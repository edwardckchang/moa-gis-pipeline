import pandas as pd
from vincenty import vincenty
from logs_handle import logger
from rasterio.mask import mask
import re
import numpy as np
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
import geopandas as gpd
import os
from shapely.geometry import Polygon
import cv2

def natural_sort_key(s):
    """
    用於自然排序的 key 函式。
    將字串分割成字母和數字部分，並將數字部分轉換為整數。
    """
    if pd.isna(s):
        # 處理 NaN/None 值，確保它們不會導致錯誤
        return ('', 0)
        
    # 確保輸入是字串
    s = str(s)
    
    parts = re.split(r'(\d+)', s)
    result = []
    for part in parts:
        if part.isdigit():
            result.append(int(part))
        else:
            result.append(part)
    return tuple(result)

def get_width_height_from_geographic_mapping(bounds, target_resolution_m=100):
    """
    根據 BBOX 和目標解析度 (target_resolution_m/px) 計算 WxH 像素數。
    用戶在此處應使用 Vincenty/測地線方法進行精確計算。
    
    Args:
        bounds (tuple): (lon_min, lat_min, lon_max, lat_max)
        target_resolution_m (int): 每個像素的目標邊長 (公尺)。
    
    Returns:
        tuple: (W_pixel, H_pixel)
    """
    mid_longitude = (bounds[0] + bounds[2]) / 2
    mid_latitude = (bounds[1] + bounds[3]) / 2
    logger.info(f"mid point: {mid_longitude} {mid_latitude}")
    longitude_start = (mid_latitude, bounds[0])  # (Lat, Lon)
    longitude_end = (mid_latitude, bounds[2])    # (Lat, Lon)
    latitude_start = (bounds[1], mid_longitude)  # (Lat, Lon)
    latitude_end = (bounds[3], mid_longitude)    # (Lat, Lon)
    distance_longitude = vincenty(longitude_start, longitude_end)
    distance_latitude = vincenty(latitude_start, latitude_end)
    W_pixel = round(distance_longitude * 1000 / target_resolution_m)
    H_pixel = round(distance_latitude * 1000 / target_resolution_m)
    return max(1, int(W_pixel)), max(1, int(H_pixel))


def png_geographic_mapping(polygon, bounds, png_path):
    """
    執行 PNG 圖片的地理遮罩。通過手動設置 Affine 轉換矩陣，
    將缺乏地理參考的 WMS 圖片與 Shapely 多邊形精確對齊。

    Args:
        polygon (shapely.geometry.Polygon): 鄉鎮的 Shapely 多邊形幾何物件。
        bounds (tuple): 鄉鎮的 BBOX (lon_min, lat_min, lon_max, lat_max)。
        png_path (str): WMS 下載的 PNG 圖片路徑。

    Returns:
        tuple: (out_img, out_transform)
            out_img (numpy.ndarray): 遮罩後的影像數據 (只有多邊形內有資訊，外部為 255)。
            out_transform (affine.Affine): 遮罩後影像的新地理參考轉換矩陣。
        None, None: 如果執行失敗。
    """
    # BBOX = [lon_min, lat_min, lon_max, lat_max]
    BBOX = [bounds[0], bounds[1], bounds[2], bounds[3]]
    W_pixel, H_pixel = get_width_height_from_geographic_mapping(bounds)
    # 讀取 PNG 圖片
    try:
        # 使用 Image.open 讀取 PNG，並確保轉換為 RGB 格式
        png_data_array = np.array(Image.open(png_path).convert("RGB"))
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {png_path}")
        return None, None
    except Exception as e:
        print(f"讀取圖片時發生錯誤: {e}")
        return None, None
    # 檢查 PNG 矩陣尺寸是否與計算的 WxH 匹配
    if png_data_array.shape[0] != H_pixel or png_data_array.shape[1] != W_pixel:
        # 此處應包含錯誤處理，但在實際應用中，WMS 應返回計算尺寸
        print(f"注意：下載圖片尺寸 {png_data_array.shape[1]}x{png_data_array.shape[0]} 與計算尺寸 {W_pixel}x{H_pixel} 不匹配。")
        # 為了演示，我們使用實際圖片尺寸進行遮罩，但這可能犧牲了地理精度
        H_pixel, W_pixel = png_data_array.shape[0], png_data_array.shape[1]
    # 轉換 Shapely 物件為 GeoJSON 字典格式 (rasterio.mask 需要) 問題的核心：GeoPandas 幾何要轉成 GeoJSON 字典
    target_polygon_geojson = [polygon.__geo_interface__]

    # 1. 計算 Affine 轉換矩陣
    # 這將 BBOX 的地理範圍映射到 WxH 的像素矩陣上
    transform = from_bounds(
        west=BBOX[0], south=BBOX[1], east=BBOX[2], north=BBOX[3], 
        width=W_pixel, height=H_pixel
    )

    # 2. 在記憶體中創建一個具備地理參考的虛擬檔案
    with MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            count=png_data_array.shape[2], # 3 個波段 (RGB)
            height=H_pixel, 
            width=W_pixel, 
            dtype=png_data_array.dtype,
            crs='EPSG:4326', # WMS BBOX 通常是 WGS84
            transform=transform # 核心地理參考
        ) as dst:
            # 將 PNG 數組寫入虛擬檔案 (PIL/Numpy 的 HWC -> Rasterio 的 CHW)
            dst.write(png_data_array.transpose((2, 0, 1))) 
            
            # 3. 執行遮罩操作
            out_img, out_transform = mask(
                dataset=dst, 
                shapes=target_polygon_geojson,  # GeoJSON 格式的多邊形
                crop=False, # 不裁剪到最小外框，保持整個 BBOX 範圍的輸出
                filled=True,
                nodata=255 # 塗白
            )
            
            # 返回遮罩後的數據和新的 transform
            return out_img, out_transform
        
def shp_reader(shp_path):
    shp_path_nor = os.path.normpath(shp_path)
    gdf = gpd.read_file(shp_path_nor)
    gdf['sort_key'] = gdf['TOWNID'].apply(natural_sort_key)
    gdf_sorted = gdf.sort_values(by='sort_key', ascending=True).reset_index(drop=True)
    return gdf_sorted

shp_path = os.path.join("GIS", 
                        "Township(town_city_district)boundaries", 
                        "TOWN_MOI_1140318.shp")

try:
    if os.path.exists(shp_path):
        gdf = gpd.read_file(shp_path)
        gdf['sort_key'] = gdf['TOWNID'].apply(natural_sort_key)
        gdf_sorted = gdf.sort_values(by='sort_key', ascending=True).reset_index(drop=True)
        print(gdf_sorted.head(10))
        # print(gdf.head())
except Exception as e:
    print(e)
