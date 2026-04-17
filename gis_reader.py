"""
gis_reader.py
=============
負責 GIS 地理資料的讀取、排序與空間運算。

職責範圍：
  - SHP 檔案讀取（geopandas）
  - TOWNID 自然排序
  - 依 BBOX 計算 WMS 請求所需的像素寬高（vincenty 測地線）
  - 以 Shapely Polygon 對 numpy 影像陣列執行 rasterio 地理遮罩

設計原則：
  - 所有函式為無狀態的純函式，不持有模組層級的全域狀態
  - 影像以記憶體內的 numpy.ndarray 傳遞，不落地至磁碟
  - 座標系統預設 EPSG:4326（WGS84）；面積計算請在呼叫端轉換為 EPSG:3826

TODO（Phase 2 — 層級一）：
  - [ ] shp_reader() 目前硬寫 TOWNID 排序欄位，待確認 COUNTY SHP 的排序欄位名稱後擴充
  - [ ] shp_reader() 回傳的 GeoDataFrame 尚未做 CRS 驗證，入庫前需確認為 EPSG:4326
  - [ ] png_geographic_mapping() 尺寸不匹配時目前直接回傳 (None, None)，
        Phase 3 可評估加入容差容錯邏輯（例如誤差 ±1px 時仍繼續）

相依模組：
  geopandas, rasterio, shapely, vincenty, pandas, numpy
  logs_handle（統一 logger）
"""

import pandas as pd
from vincenty import vincenty
from logs_handle import logger
from rasterio.mask import mask
import re
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
import geopandas as gpd
import os
from utils import Checkpoint

def _natural_sort_key(s):
    """
    將字串轉換為自然排序 key，使數字部分按數值大小排序。

    處理邏輯：以正則將字串分割為「文字段」與「數字段」交替組成的 tuple，
    數字段轉為 int，確保 "TOWN2" < "TOWN10"（字典序會得出相反結果）。
    NaN / None 值一律排至最前（回傳 ('', 0)）。

    Args:
        s: 待轉換的字串，或可能為 NaN/None 的值。

    Returns:
        tuple: 可用於 sort() / sorted() key 的排序 tuple。

    Examples:
        >>> sorted(["TOWN10", "TOWN2", "TOWN1"], key=_natural_sort_key)
        ['TOWN1', 'TOWN2', 'TOWN10']
    """
    if pd.isna(s):
        return ('', 0)

    s = str(s)
    parts = re.split(r'(\d+)', s)
    result = []
    for part in parts:
        if part.isdigit():
            result.append(int(part))
        else:
            result.append(part)
    return tuple(result)


def get_width_height_from_geographic_mapping(
    bounds: tuple,
    target_resolution_m: int = 100
) -> tuple:
    """
    依 BBOX 與目標地面解析度（公尺/像素），以 Vincenty 測地線公式計算
    WMS 請求所需的影像像素尺寸（width × height）。

    計算策略：
      - 水平距離（longitude 方向）：固定中心緯度，量測西東兩端的大圓距離
      - 垂直距離（latitude 方向）：固定中心經度，量測南北兩端的大圓距離
      - 兩距離分別除以 target_resolution_m，四捨五入取整

    Args:
        bounds (tuple): (lon_min, lat_min, lon_max, lat_max)，EPSG:4326。
        target_resolution_m (int): 每像素對應的地面邊長（公尺），預設 100m/px。

    Returns:
        tuple[int, int]: (W_pixel, H_pixel)，最小值各為 1。

    Notes:
        - vincenty() 回傳單位為公里，需 × 1000 後再除以 target_resolution_m
        - 台灣全島 BBOX (119.3, 21.9, 122.1, 25.4) 在 100m/px 下
          實測結果約為 (2857, 3876)，非直覺的 (311, 389)，
          原因是 BBOX 範圍遠大於台灣陸地面積

    Examples:
        >>> get_width_height_from_geographic_mapping((119.3, 21.9, 122.1, 25.4), 100)
        (2857, 3876)
    """
    mid_longitude = (bounds[0] + bounds[2]) / 2
    mid_latitude = (bounds[1] + bounds[3]) / 2
    logger.notice(f"BBOX 中心點：lon={mid_longitude:.4f}, lat={mid_latitude:.4f}")

    longitude_start = (mid_latitude, bounds[0])  # (Lat, Lon)
    longitude_end   = (mid_latitude, bounds[2])  # (Lat, Lon)
    latitude_start  = (bounds[1], mid_longitude) # (Lat, Lon)
    latitude_end    = (bounds[3], mid_longitude) # (Lat, Lon)

    distance_longitude = vincenty(longitude_start, longitude_end)  # km
    distance_latitude  = vincenty(latitude_start,  latitude_end)   # km

    W_pixel = round(distance_longitude * 1000 / target_resolution_m)
    H_pixel = round(distance_latitude  * 1000 / target_resolution_m)

    logger.notice(f"BBOX：W={W_pixel}px H={H_pixel}px（{target_resolution_m}m/px）")
    return max(1, int(W_pixel)), max(1, int(H_pixel))


def png_geographic_mapping(
    polygon,
    bounds: tuple,
    image_array,
    target_res: int = 100
) -> tuple:
    """
    以 Shapely Polygon 對 WMS 下載影像執行 rasterio 地理遮罩（不落地）。

    流程：
      1. 依 bounds 計算預期尺寸，與 image_array 實際尺寸嚴格比對
      2. 以 from_bounds() 建立仿射轉換（affine transform），賦予影像地理參考
      3. 在記憶體中建立暫存 GTiff（MemoryFile），執行 rasterio.mask()
      4. crop=False：保留原始尺寸（不裁切），多邊形外填 nodata=255（白色背景）

    Args:
        polygon: Shapely Polygon 或 MultiPolygon，EPSG:4326，作為遮罩形狀。
        bounds (tuple): (lon_min, lat_min, lon_max, lat_max)，對應 image_array 的地理範圍。
        image_array (numpy.ndarray): WMS 下載的 BGR 影像，shape=(H, W, C)，dtype=uint8。
        target_res (int): 用於尺寸驗證的目標解析度（公尺/像素），預設 100。

    Returns:
        tuple[numpy.ndarray, affine.Affine] | tuple[None, None]:
            - 成功：(out_img, out_transform)
              out_img shape = (C, H, W)，多邊形外像素值為 255
            - 失敗（尺寸不符或遮罩運算錯誤）：(None, None)

    Notes:
        - image_array 為 cv2 讀取的 BGR 格式（channel-last），寫入 GTiff 前
          需 transpose((2, 0, 1)) 轉為 channel-first
        - 回傳的 out_img 同樣為 channel-first (C, H, W)，下游使用時需注意

    TODO（Phase 3）：
        - 評估尺寸容差邏輯：WMS 實際回傳可能因服務端 rounding 而有 ±1px 偏差，
          目前做法是直接回傳 (None, None)，Phase 3 可改為 resize 後繼續處理
    """
    if image_array is None:
        return None, None

    # 1. 計算預期尺寸並驗證
    W_expected, H_expected = get_width_height_from_geographic_mapping(bounds, target_res)
    H_actual, W_actual = image_array.shape[:2]

    if W_actual != W_expected or H_actual != H_expected:
        logger.warning(
            f"下載圖片尺寸 {W_actual}×{H_actual} 與計算尺寸 {W_expected}×{H_expected} 不匹配，"
            f"略過此影像。"
        )
        return None, None

    # 2. 轉換 Shapely 物件為 GeoJSON 字典格式（rasterio.mask 所需）
    target_polygon_geojson = [polygon.__geo_interface__]

    # 3. 建立仿射轉換（affine transform）
    transform = from_bounds(
        west=bounds[0], south=bounds[1],
        east=bounds[2], north=bounds[3],
        width=W_actual, height=H_actual
    )

    try:
        with MemoryFile() as memfile:
            with memfile.open(
                driver='GTiff',
                count=image_array.shape[2],
                height=H_actual,
                width=W_actual,
                dtype=image_array.dtype,
                crs='EPSG:4326',
                transform=transform
            ) as dst:
                dst.write(image_array.transpose((2, 0, 1)))  # (H,W,C) → (C,H,W)

                out_img, out_transform = mask(
                    dataset=dst,
                    shapes=target_polygon_geojson,
                    crop=False,    # 保留原始尺寸
                    filled=True,
                    nodata=255     # 多邊形外填白（與 WMS 背景一致）
                )
                return out_img, out_transform

    except Exception as e:
        logger.error(f"遮罩運算失敗: {e}")
        return None, None


def shp_reader(shp_path: str) -> gpd.GeoDataFrame:
    """
    讀取 SHP 檔案並依 TOWNID 欄位自然排序後回傳。

    處理步驟：
      1. os.path.normpath() 標準化路徑（相容 Windows / Linux）
      2. geopandas.read_file() 讀取 SHP，自動解析 geometry
      3. 依 TOWNID 欄位以 _natural_sort_key() 排序（數字部分按數值大小）
      4. 刪除暫存排序欄位後回傳乾淨的 GeoDataFrame

    Args:
        shp_path (str): SHP 檔案路徑，例如 "GIS/boundaries/TOWN_MOI_1140318.shp"。

    Returns:
        geopandas.GeoDataFrame: 依 TOWNID 自然排序、重設 index 的 GeoDataFrame。
            geometry 欄位保留 geopandas 原生格式（Shapely 物件），
            CRS 由 SHP 附帶的 .prj 自動載入。

    Raises:
        FileNotFoundError: 若 shp_path 不存在（由 geopandas.read_file() 拋出）。
        Exception: 其他讀取或排序錯誤（呼叫端自行處理）。

    TODO（Phase 2）：
        - [ ] 加入 CRS 驗證：若 gdf.crs != EPSG:4326 則執行 gdf.to_crs("EPSG:4326")
    """
    shp_path_nor = os.path.normpath(shp_path)# --- 自動判定排序欄位 ---
    # 使用 lower() 判斷路徑中是否包含 'county'
    if "geometry" not in gdf.columns:
        logger.warning(f"geometry 欄位不存在於 {shp_path_nor}，不是正確的 SHP檔案，將回傳空白 GeoDataFrame。")
        return gpd.GeoDataFrame()
    if "TOWNID".lower() not in gdf.columns.str.lower() or "COUNTYID".lower() not in gdf.columns.str.lower():
        logger.warning(f"排序欄位 TOWNID 或 COUNTYID 不存在於 {shp_path_nor}，將回傳原始 GeoDataFrame。")
        return gdf
    if 'county' in shp_path_nor.lower():
        sort_col = "COUNTYID"
    else:
        sort_col = "TOWNID"
    gdf = gpd.read_file(shp_path_nor)

    if sort_col not in gdf.columns:
        logger.warning(f"預期的排序欄位 {sort_col} 不存在於 {shp_path_nor}，將跳過排序。")
        return gdf

    gdf['sort_key'] = gdf[sort_col].apply(_natural_sort_key)
    gdf_sorted = gdf.sort_values(by='sort_key', ascending=True).reset_index(drop=True)
    gdf_sorted.drop(columns=['sort_key'], inplace=True)

    logger.info(f"SHP 讀取完成：{shp_path_nor}，共 {len(gdf_sorted)} 筆，CRS={gdf_sorted.crs}")
    return gdf_sorted


if __name__ == '__main__':
    shp_path = os.path.join("GIS", "TOWN_MOI_1140318.shp")

    try:
        if os.path.exists(shp_path):
            gdf = shp_reader(shp_path)
            print(gdf.head(10))
    except Exception as e:
        print(e)

    w, h = get_width_height_from_geographic_mapping(
        bounds=(119.3, 21.9, 122.1, 25.4),
        target_resolution_m=100
    )
    print(w, h)
