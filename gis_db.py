import geopandas as gpd
from file_utils import load_json_data, save_json_data
import numpy as np
import cv2
import os
from gis_quantifier import load_image_with_chinese_path, decode_png_color_value
from logs_handle import logger
from database_manager import table_exists, execute_sql

def get_count(conn, table_id):
    if table_exists(conn, table_id):
        # 這裡使用你底層的 SQL 執行工具
        res = execute_sql(conn, f'SELECT COUNT(*) FROM "{table_id}";', fetch_all=True)
        return f"{res[0]['count'] if res else 0}"
    return "0"

def infer_schema_from_geodataframe(gdf: gpd.GeoDataFrame) -> dict:
    """
    從 GeoDataFrame 推斷 metadata_schema。
    """
    PANDAS_TO_SCHEMA = {
        "int64":         "integer",
        "float64":       "float",
        "bool":          "boolean",
        "datetime64[ns]":"datetime",
        "object":        "text",
        "geometry":      "geometry",
    }
    schema = {}
    for col, dtype in gdf.dtypes.items():
        if col == "foreign_key":
            continue
        dtype_str = str(dtype)
        schema[col] = PANDAS_TO_SCHEMA.get(dtype_str, "text")
    return schema

def add_color_mapping_level1(illustrative_diagram_name: str, unique_colors, metadata_path):
    """
    讓使用者輸入圖片中每種(非白色的)顏色的意義，並儲存到 metadata 中。此為level 1標籤。
    
    Args:
        png_file_name (str): 圖例說明的檔名。
        unique_colors (List[tuple]): 圖片中所有不重複且非白色的 BGR 顏色列表 [(B, G, R), ...]。
    """
    
    color_blocks = []
    block_size = 20
    
    for b, g, r in unique_colors:
        color_block = np.full((block_size, block_size, 3), 
                              (b, g, r), 
                              dtype=np.uint8)
        color_blocks.append(color_block)

    if not color_blocks:
        logger.error("未找到任何有效色彩值來建立顏色塊。")
        return# 1. 建立顏色塊 (邏輯不變)
    color_blocks = []
    block_size = 30
    for b, g, r in unique_colors:
        color_block = np.full((block_size, block_size, 3), (b, g, r), dtype=np.uint8) 
        color_blocks.append(color_block)

    display_image = np.hstack(color_blocks)
    
    # 2. 顯示視窗並進入互動模式
    window_name = f'Mapping: {illustrative_diagram_name} Unique Colors (Non-White BGR order)'
    cv2.imshow(window_name, display_image)
    
    print("\n----------------------------------------------------")
    print(f"請參照視窗 [{window_name}] 中的顏色塊，進行後續輸入。")
    
    # 這裡加入一個短暫的等待，確保視窗彈出
    cv2.waitKey(100) 

    # --- 互動輸入階段 ---
    
    # 使用迴圈讓 cv2 視窗保持活躍
    def get_input_while_showing(prompt):
        """在等待 input() 時，保持 cv2 視窗活躍的輔助函式"""
        # 設置一個小的等待時間，讓視窗有時間刷新
        cv2.waitKey(1) 
        return input(prompt)

    # 3. 載入並檢查 Metadata
    metadata = load_json_data(metadata_path)
    if not metadata:
        metadata = {}
        existing_entry = None
    else:
        existing_entry = metadata.get(illustrative_diagram_name)
    
    if existing_entry is not None:
        logger.warning(f"警告: {illustrative_diagram_name} 的資料已存在。請手動修改或重新執行。")
        print("已輸入的資料如下:")
        print("data_type:")
        print(metadata[illustrative_diagram_name].get("data_type"))
        print("color_to_value:")
        print(metadata[illustrative_diagram_name].get("color_to_value"))
        input("按 Enter 關閉視窗...")
        cv2.destroyAllWindows() # 完成後關閉視窗
        return
    
    # 4. 互動式輸入 - 資料類型
    print(f"\n--- 開始為檔案: {illustrative_diagram_name} 輸入顏色對應資訊 ---")
    data_type = get_input_while_showing("請輸入此地圖資料的類型 (例如: '分級', 'pH連續分段', '土質'): ")
    from utils import process_string
    data_type = process_string(data_type)
    print(f"您輸入的資料類型為: {data_type}")
    
    color_map = {}
    
    # 5. 互動式輸入 - 顏色對應
    for b, g, r in unique_colors:
        color_key = f"{b},{g},{r}"
        
        # 讓使用者輸入這個顏色的意義/等級
        prompt = f"請輸入 BGR {color_key} ({b}, {g}, {r}) 代表的意義或等級: "
        value = get_input_while_showing(prompt)
        from utils import process_string
        value = process_string(value)
        
        color_map[color_key] = value
        print(f"您輸入色彩值 {color_key} 的定義值為: {value}")
    
    # 6. 建立新的資料結構並儲存
    new_entry = {
        "data_type": data_type,
        "color_to_value": color_map
    }
    
    metadata[illustrative_diagram_name] = new_entry
         
    saved_path = save_json_data(metadata, metadata_path)
    if saved_path:
        logger.info(f"\n--- 成功儲存資料到 {saved_path} ---")
    else:
        logger.error(f"\n--- 儲存資料到 {metadata_path} 時發生錯誤。 ---")


    # 7. 完成輸入，關閉視窗
    print("--- 顏色映射輸入完成，正在關閉視窗 ---")
    cv2.destroyAllWindows()

if __name__ == '__main__':
# 預設儲存 metadata 的 JSON 檔案名稱
    metadata_file = 'geographic_color_metadata.json'
    metadata_path = os.path.join("farmland_spatial_map", metadata_file)
    png_path = r"C:\Python\work\farmland_spatial_map\soil_survey\母岩性質.png"
    illustrative_diagram_name = "Parent Material Property(illustrative_diagram)"
    from utils import process_string
    illustrative_diagram_name = process_string(illustrative_diagram_name)
    png_path = os.path.normpath(png_path)
    # print(png_path)
    try:
        if os.path.exists(png_path):
            png = load_image_with_chinese_path(png_path)
            print("圖片形狀: ", png.shape)
            unique_colors = decode_png_color_value(png)
            if unique_colors is not None:
                add_color_mapping_level1(illustrative_diagram_name, unique_colors)
        else:
            logger.error(f"找不到檔案: {png_path}")
    except Exception as e:
        logger.error(e)