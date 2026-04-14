"""
gis_db.py
=========
GIS 專屬資料庫操作模組，集中管理所有 GIS 相關的 SQL 邏輯。

職責範圍：
  - GIS_001（行政界線）的資料表建立、寫入、查詢
  - GIS_metadata（執行紀錄）的資料表建立與寫入
  - GeoDataFrame schema 推斷（供 database_manager.table_columns_sql() 使用）
  - geographic_color_metadata.json 的互動式維護工具

設計原則：
  - 業務邏輯層（main_gis.py）不直接寫 SQL，統一經過此模組
  - conn 物件由呼叫端（main_gis.py）持有並傳入，此模組不管理連線生命週期
  - 所有資料表名稱以常數定義，避免字串散落各處

【架構決策紀錄】geographic_color_metadata.json 不入庫：
  色彩對應表（color_to_value）為異質結構——不同圖層的 value 可以是
  int、float、str 或 list，強行入庫需要 JSONB 或多態設計，複雜度高、
  收益低（查詢需求少，且 git 已提供完整的版本追蹤能力）。
  因此決策：此檔案以 JSON 管理並納入 git 追蹤，不存入 PostgreSQL。
  更新時將舊版重命名為 geographic_color_metadata.{YYYYMMDD}.bak 保留備份。
  相關讀取邏輯集中在 add_color_mapping_level1() 與 gis_quantifier.py。

TODO（Phase 2 — 層級一）：
  - [ ] 定義資料表名稱常數：
        TABLE_GIS_001 = "GIS_001"
        TABLE_GIS_METADATA = "GIS_metadata"
  - [ ] 實作 create_gis_001_table(conn)：
        建立 GIS_001 資料表，欄位由 infer_schema_from_geodataframe() 推斷，
        PRIMARY KEY 為 TOWNID（文字型，來自 SHP 欄位）
  - [ ] 實作 upsert_gis_001(conn, record)：
        將單筆行政界線資料寫入 GIS_001，
        conflict 欄位為 TOWNID，geometry 以 WKT 文字存入
  - [ ] 實作 get_all_towns(conn) -> list[dict]：
        從 GIS_001 讀取全部鄉鎮資料，供 geographic_mapping() 的迴圈使用。
        每筆 dict 至少需包含：
          - "TOWNID"：鄉鎮代碼
          - "TOWNNAME"：鄉鎮中文名稱（供查中英對照表用）
          - "geometry_wkt"：WKT 字串（呼叫端還原為 Shapely 物件）
        呼叫端以 shapely.wkt.loads(row["geometry_wkt"]) 還原，
        再取 polygon.bounds 計算 BBOX
  - [ ] 實作 create_gis_metadata_table(conn)：
        建立 GIS_metadata 資料表，記錄每次執行的影像儲存紀錄，
        schema 設計見下方 log_gis_metadata() TODO

TODO（Phase 3 — 層級二）：
  - [ ] 實作 log_gis_metadata(conn, record)：
        寫入單筆影像處理執行紀錄，record 應包含：
          - "townid"：對應 GIS_001 的 TOWNID（外鍵）
          - "classification"：圖層英文名稱
          - "shp_version"：SHP 檔名（例如 "TOWN_MOI_1140318"），作為可追溯的界線版本
          - "stage"："raw" 或 "masked"
          - "status"：save_image() 回傳的 status（"created" | "updated" | "unchanged" | "error"）
          - "file_path"：影像的完整落地路徑（save_image() 回傳的 path）
          - "recorded_at"：寫入時間戳記（TIMESTAMP WITH TIME ZONE DEFAULT NOW()）
        conflict 欄位建議為 (townid, classification, shp_version, stage)，
        status 與 file_path 於衝突時更新

相依模組：
  geopandas, numpy, cv2（OpenCV）
  file_utils（load_json_data, save_json_data）
  gis_quantifier（load_image_with_chinese_path, decode_png_color_value）
  database_manager（table_exists, execute_sql, table_columns_sql）
  logs_handle（統一 logger）
"""

import geopandas as gpd
from file_utils import load_json_data, save_json_data
import numpy as np
import cv2
import os
from gis_quantifier import load_image_with_chinese_path, decode_png_color_value
from logs_handle import logger
from database_manager import table_exists, execute_sql, table_columns_sql


def get_count(conn, table_id: str) -> str:
    """
    查詢指定資料表的列數，若資料表不存在則回傳 "0"。

    主要用於 CLI 選單顯示各資料表的目前筆數，提供使用者入庫狀態的快速確認。

    Args:
        conn: 使用中的 psycopg2 connection 物件。
        table_id (str): 資料表名稱（不含引號），例如 "GIS_001"。

    Returns:
        str: 資料表列數的字串表示，例如 "368"；若資料表不存在則回傳 "0"。

    Notes:
        - 回傳型別為 str 而非 int，方便直接用於 CLI 字串拼接顯示
        - execute_sql() 搭配 fetch_all=True 時，COUNT(*) 結果在 dict 的 key 為 'count'

    TODO（Phase 2）：
        - [ ] 確認 PostgreSQL COUNT(*) 回傳的欄位名稱是否確實為 'count'（小寫），
              或需改用 fetch_one=True 直接取 res[0]
    """
    if table_exists(conn, table_id):
        res = execute_sql(conn, f'SELECT COUNT(*) FROM "{table_id}";', fetch_all=True)
        return f"{res[0]['count'] if res else 0}"
    return "0"


def infer_schema_from_geodataframe(gdf: gpd.GeoDataFrame) -> dict:
    """
    從 GeoDataFrame 的欄位型別推斷 metadata_schema，
    供 database_manager.table_columns_sql() 轉換為 PostgreSQL 欄位定義使用。

    型別對應規則（pandas dtype → GIS_TYPE_MAP 標籤）：
      int64         → "integer"   （對應 BIGINT）
      float64       → "float"     （對應 DOUBLE PRECISION）
      bool          → "boolean"
      datetime64[ns]→ "datetime"  （對應 TIMESTAMP WITH TIME ZONE）
      object        → "text"
      geometry      → "geometry"  （對應 geometry(Geometry, 4326)，需 PostGIS）
      其他未知型別   → "text"（保守退回）

    Args:
        gdf (geopandas.GeoDataFrame): 來源 GeoDataFrame，通常為 shp_reader() 的回傳值。

    Returns:
        dict[str, str]: 欄位名稱 → 型別標籤的對應字典，
            例如 {"TOWNID": "text", "geometry": "geometry", "AREA": "float"}。
            "foreign_key" 欄位一律跳過（由呼叫端另行處理）。

    Notes:
        - 回傳的 schema dict 直接傳入 database_manager.table_columns_sql()
          即可產生 CREATE TABLE 所需的欄位定義字串
        - SHP 的字串欄位在 geopandas 中通常為 object dtype，故對應到 "text"
        - geometry 欄位在 gdf.dtypes 中的型別字串為 "geometry"（geopandas 自訂型別）

    TODO（Phase 2）：
        - [ ] 驗證 geopandas geometry 欄位的 dtype.__str__() 是否確實為 "geometry"，
              不同版本的 geopandas 行為可能略有差異
    """
    PANDAS_TO_SCHEMA = {
        "int64":          "integer",
        "float64":        "float",
        "bool":           "boolean",
        "datetime64[ns]": "datetime",
        "object":         "text",
        "geometry":       "geometry",
    }
    schema = {}
    for col, dtype in gdf.dtypes.items():
        if col == "foreign_key":
            continue
        dtype_str = str(dtype)
        mapped = PANDAS_TO_SCHEMA.get(dtype_str, "text")
        if mapped == "text" and dtype_str not in PANDAS_TO_SCHEMA:
            logger.warning(f"infer_schema_from_geodataframe：未知 dtype '{dtype_str}'（欄位：'{col}'），退回 text。")
        schema[col] = mapped
    return schema


def add_color_mapping_level1(
    illustrative_diagram_name: str,
    unique_colors: list,
    metadata_path: str
) -> None:
    """
    互動式 CLI 工具：為指定圖層的每種像素色彩輸入對應的業務意義，
    並將結果寫入 geographic_color_metadata.json。此為「層級一標籤」建立流程。

    使用情境：
      首次處理新的 WMS 圖層時，由開發者手動執行一次，建立該圖層的色彩對應表。
      建立完成後，量化流程（gis_quantifier.py）即可依此 JSON 進行像素統計。

    流程：
      1. 依 unique_colors 建立 30×30px 色塊並以 cv2.imshow() 顯示，
         方便對照目視識別每種顏色
      2. 載入 metadata_path 的 JSON；若該圖層名稱已存在則提示並中止，
         防止意外覆蓋已建立的對應表
      3. 互動式輸入資料類型（data_type）與每種色彩的對應值（color_map）
      4. 將新條目寫回 JSON 並關閉 cv2 視窗

    Args:
        illustrative_diagram_name (str): 圖層的唯一識別名稱，作為 JSON 的頂層 key，
            建議格式："{layer_name}[illustrative_diagram]"，
            例如 "soil_ph[illustrative_diagram]"。
            應先以 utils.process_string() 標準化（小寫、空格轉底線）後再傳入。
        unique_colors (list[tuple]): 圖片中所有不重複且非白色的 BGR 色彩列表，
            由 gis_quantifier.decode_png_color_value() 產生，
            格式為 [(B, G, R), ...]，各分量為 int。
        metadata_path (str): geographic_color_metadata.json 的檔案路徑。

    Returns:
        None。結果直接寫入 metadata_path 所指的 JSON 檔案。

    Side Effects:
        - 開啟 cv2 視窗（執行期間）
        - 修改 metadata_path 指向的 JSON 檔案
        - 透過 input() 阻塞等待使用者輸入

    Notes:
        - 此函式為一次性的手動維護工具，不應在自動化流程（排程）中呼叫
        - color_map 的 value 目前存為使用者輸入的原始字串；
          若需要數值型別（int/float/list），需在輸入後另行轉換
        - JSON key 格式為 "B,G,R"（逗號分隔字串），例如 "1,254,3"
        - 更新既有 JSON 時，舊版本會被重命名為
          geographic_color_metadata.{YYYYMMDD}.bak 保留備份（TODO 待實作）

    TODO（Phase 3）：
        - [ ] 考慮支援多值輸入（例如 soil_texture 的 value 為 [sand%, silt%, clay%] 列表），
              目前僅儲存單一字串，複雜結構需手動編輯 JSON
        - [ ] 若 unique_colors 數量超過一定閾值（例如 20 種），
              考慮分頁顯示色塊，避免橫向視窗過寬
    """
    if not unique_colors:
        logger.error("add_color_mapping_level1：unique_colors 為空，無法執行。")
        return

    # 1. 建立色塊橫向拼接影像並顯示
    block_size = 30
    color_blocks = [
        np.full((block_size, block_size, 3), (b, g, r), dtype=np.uint8)
        for b, g, r in unique_colors
    ]
    display_image = np.hstack(color_blocks)

    window_name = f'Mapping: {illustrative_diagram_name} Unique Colors (Non-White BGR order)'
    cv2.imshow(window_name, display_image)

    print("\n----------------------------------------------------")
    print(f"請參照視窗 [{window_name}] 中的顏色塊，進行後續輸入。")
    cv2.waitKey(100)  # 確保視窗彈出後再繼續

    def get_input_while_showing(prompt: str) -> str:
        """在等待 input() 時，保持 cv2 視窗活躍。"""
        cv2.waitKey(1)
        return input(prompt)

    # 2. 載入並檢查既有 Metadata，防止覆蓋
    metadata = load_json_data(metadata_path)
    if not metadata:
        metadata = {}

    existing_entry = metadata.get(illustrative_diagram_name)
    if existing_entry is not None:
        logger.warning(f"{illustrative_diagram_name} 的色彩對應資料已存在，請手動修改 JSON 或重新執行。")
        print("已存在的資料如下：")
        print(f"  data_type: {existing_entry.get('data_type')}")
        print(f"  color_to_value: {existing_entry.get('color_to_value')}")
        input("按 Enter 關閉視窗...")
        cv2.destroyAllWindows()
        return

    # 3. 互動式輸入 — 資料類型
    print(f"\n--- 開始為圖層 [{illustrative_diagram_name}] 輸入顏色對應資訊 ---")
    from utils import process_string
    data_type = process_string(
        get_input_while_showing("請輸入此地圖資料的類型 (例如: '分級', 'pH連續分段', '土質'): ")
    )
    print(f"您輸入的資料類型為: {data_type}")

    # 4. 互動式輸入 — 逐色對應
    color_map = {}
    for b, g, r in unique_colors:
        color_key = f"{b},{g},{r}"
        value = process_string(
            get_input_while_showing(f"請輸入 BGR {color_key} 代表的意義或等級: ")
        )
        color_map[color_key] = value
        print(f"  色彩 {color_key} → {value}")

    # 5. 寫回 JSON
    metadata[illustrative_diagram_name] = {
        "data_type": data_type,
        "color_to_value": color_map
    }

    saved_path = save_json_data(metadata, metadata_path)
    if saved_path:
        logger.info(f"色彩對應表已儲存至 {saved_path}")
    else:
        logger.error(f"儲存色彩對應表至 {metadata_path} 時發生錯誤。")

    # 6. 關閉視窗
    print("--- 顏色映射輸入完成，正在關閉視窗 ---")
    cv2.destroyAllWindows()

# TODO（Phase 2）：實作 get_all_towns(conn) -> list[dict]
# TODO（Phase 2）：實作 create_gis_metadata_table(conn) -> None
# TODO（Phase 3）：實作 log_gis_metadata(conn, record: dict) -> bool

def create_gis_polygon_table(conn) -> bool:
    """
    若 GIS_001 資料表不存在，則依固定 schema 建立之。
 
    GIS_001 的欄位結構固定對應 TOWN_MOI SHP 的標準欄位，
    不使用 infer_schema_from_geodataframe() 動態推斷——
    原因：SHP 欄位在不同版本的 MOI 發布中偶有增刪，
    固定 schema 可確保資料表結構穩定，新版 SHP 若有新欄位
    由 ensure_columns_exist() 在入庫時補充。
 
    資料表 schema：
        TOWNID      TEXT        PRIMARY KEY   鄉鎮代碼，例如 "10016010"
        COUNTYID    TEXT                      縣市代碼，例如 "10016"
        TOWNNAME    TEXT                      鄉鎮中文名稱，例如 "大安區"
        COUNTYNAME  TEXT                      縣市中文名稱，例如 "臺中市"
        geometry    geometry(Geometry, 4326)  PostGIS geometry 欄位
        shp_version TEXT                      來源 SHP 檔名，例如 "TOWN_MOI_1140318"
        created_at  TIMESTAMP WITH TIME ZONE  DEFAULT NOW()
 
    Args:
        conn: 使用中的 psycopg2 connection 物件。
 
    Returns:
        bool: 建立成功（或資料表已存在）回傳 True；失敗回傳 False。
 
    Notes:
        - geometry 欄位使用 PostGIS 原生型別，需確認 PostGIS extension 已安裝
        - TOWNID 為 PRIMARY KEY，upsert 時以此為 conflict 欄位
        - shp_version 欄位記錄本筆資料來自哪個版本的 SHP，
          行政界線更新時可依此欄位識別哪些鄉鎮是新版本寫入
 
    TODO（Phase 2）：
        - [ ] 實作本函式：
              1. 若 table_exists(conn, TABLE_GIS_001) 則直接回傳 True
              2. 組裝 CREATE TABLE SQL（參考上方 schema）
              3. 呼叫 execute_sql(conn, sql) 執行
              4. 再次 table_exists() 確認建立成功
        - [ ] 確認 PostGIS extension 是否已在目標資料庫安裝：
              SELECT 1 FROM pg_extension WHERE extname = 'postgis';
              若未安裝則 logger.error 並回傳 False
    """
    pass
 
 
def upsert_gis_polygon(conn, gdf: "gpd.GeoDataFrame", shp_version: str) -> bool:
    """
    將 shp_reader() 讀取的 GeoDataFrame 逐筆 upsert 至 GIS_001 資料表。
 
    入庫流程：
      1. 確認 GIS_001 存在（呼叫 create_gis_001_table()）
      2. 逐筆取出 gdf 的每一行（鄉鎮），組裝 record dict
      3. geometry 以 geopandas .to_wkt() 轉為 WKT 字串後存入
         （PostGIS 接受 ST_GeomFromText(wkt, 4326) 格式）
      4. 補入 shp_version 欄位
      5. 呼叫 database_manager.execute_upsert()，conflict 欄位為 TOWNID
      6. 全部完成後 logger.success()
 
    Args:
        conn: 使用中的 psycopg2 connection 物件。
        gdf (geopandas.GeoDataFrame): shp_reader() 回傳的 GeoDataFrame，
            已依 TOWNID 自然排序，CRS 為 EPSG:4326。
        shp_version (str): 來源 SHP 檔名（不含副檔名），例如 "TOWN_MOI_1140318"，
            作為可追溯的界線版本記錄於每筆資料。
 
    Returns:
        bool: 全部筆數 upsert 成功回傳 True；任一筆失敗回傳 False。
 
    Notes:
        - geometry 的 WKT 轉換：gdf.geometry[i].wkt 即可取得，
          寫入時以 ST_GeomFromText(%s, 4326) 包裝確保 CRS 正確設定
        - upsert conflict 欄位為 TOWNID；若同一 TOWNID 已存在，
          更新 geometry、shp_version、TOWNNAME、COUNTYNAME 等所有非 PK 欄位，
          意即新版界線會覆蓋舊版（舊版可由 GIS_metadata 的 shp_version 欄位追溯）
        - 建議在迴圈中以 logger.info() 記錄進度：
          f"GIS_001 寫入進度：{i+1}/{total} — {townname}"
 
    TODO（Phase 2）：
        - [ ] 實作本函式（參考上方流程）
        - [ ] 確認 geometry WKT 寫入方式：
              選項 A：欄位型別為 geometry，值以 ST_GeomFromText(%s, 4326) 包裝
              選項 B：欄位型別改為 TEXT 存 WKT，讀出時由 shapely.wkt.loads() 還原
              （選項 B 不需 PostGIS，但失去空間索引能力；依 Phase 2 實測決定）
        - [ ] 決定是否在入庫前呼叫 ensure_columns_exist()，
              處理新版 SHP 新增欄位的情況
        - [ ] 考慮加入整批失敗時的 conn.rollback()，避免部分寫入的髒資料
    """
    pass

if __name__ == '__main__':
    # 手動維護工具的使用範例：為新圖層建立色彩對應表
    # metadata_path 改為新專案根目錄下的 geographic_color_metadata.json
    metadata_path = 'geographic_color_metadata.json'

    png_path = r"C:\Python\work\farmland_spatial_map\soil_survey\母岩性質.png"
    illustrative_diagram_name = "Parent Material Property(illustrative_diagram)"

    from utils import process_string
    illustrative_diagram_name = process_string(illustrative_diagram_name)
    png_path = os.path.normpath(png_path)

    try:
        if os.path.exists(png_path):
            png = load_image_with_chinese_path(png_path)
            logger.info(f"圖片形狀: {png.shape}")
            unique_colors = decode_png_color_value(png)
            if unique_colors is not None:
                add_color_mapping_level1(illustrative_diagram_name, unique_colors, metadata_path)
        else:
            logger.error(f"找不到檔案: {png_path}")
    except Exception as e:
        logger.error(e)
