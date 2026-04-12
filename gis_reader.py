import pandas as pd
from vincenty import vincenty
from logs_handle import logger
from rasterio.mask import mask
import re
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
import geopandas as gpd
import os

def _natural_sort_key(s):
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

def png_geographic_mapping(polygon, bounds, image_array, target_res=100):
    """
    執行地理遮罩，並嚴格校驗下載影像是否符合規格。
    """

    if image_array is None: return None, None

    # 1. 【開規格】計算預期尺寸
    W_expected, H_expected = get_width_height_from_geographic_mapping(bounds, target_res)
    
    # 2. 【拿現狀】取得實際下載尺寸
    H_actual, W_actual = image_array.shape[:2]
    # 3. 【防禦性檢查】校驗尺寸是否一致
    if W_actual != W_expected or H_actual != H_expected:
        logger.warning(f"注意：下載圖片尺寸 {image_array.shape[1]}x{image_array.shape[0]} 與計算尺寸 {W_expected}x{H_expected} 不匹配。")
        return None, None
    
    # 轉換 Shapely 物件為 GeoJSON 字典格式 (rasterio.mask 需要) 問題的核心：GeoPandas 幾何要轉成 GeoJSON 字典
    target_polygon_geojson = [polygon.__geo_interface__]
    # 4. 【建立地理參考】
    # 這裡使用 W_actual (此時已確保 = W_expected) 建立對應關係
    transform = from_bounds(
        west=bounds[0], south=bounds[1], east=bounds[2], north=bounds[3], 
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
                dst.write(image_array.transpose((2, 0, 1))) 
                
                out_img, out_transform = mask(
                    dataset=dst, 
                    shapes=target_polygon_geojson,
                    crop=False, 
                    filled=True,
                    nodata=255 
                )
                return out_img, out_transform
    except Exception as e:
        logger.error(f"遮罩運算失敗: {e}")
        return None, None
        
def shp_reader(shp_path):
    shp_path_nor = os.path.normpath(shp_path)
    gdf = gpd.read_file(shp_path_nor)
    gdf['sort_key'] = gdf['TOWNID'].apply(_natural_sort_key)
    gdf_sorted = gdf.sort_values(by='sort_key', ascending=True).reset_index(drop=True)
    gdf_sorted.drop(columns=['sort_key'], inplace=True)
    return gdf_sorted

if __name__ == '__main__':
    
    shp_path = os.path.join("GIS", 
                            "TOWN_MOI_1140318.shp")

    try:
        if os.path.exists(shp_path):
            gdf = shp_reader(shp_path)
            print(gdf.head(10))
    except Exception as e:
        print(e)
    w, h =get_width_height_from_geographic_mapping(bounds=(119.3, 21.9, 122.1, 25.4), target_resolution_m=100)
    print(w, h)