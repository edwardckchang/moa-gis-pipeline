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

相依模組：
  geopandas, numpy, cv2（OpenCV）
  file_utils（load_json_data, save_json_data）
  gis_quantifier（load_image_with_chinese_path, decode_png_color_value）
  database_manager（table_exists, execute_sql, table_columns_sql）
  logs_handle（統一 logger）
"""

import geopandas as gpd
import pandas as pd
import os
from logs_handle import logger
from database_manager import table_exists, execute_sql, ensure_columns_exist
from tqdm import tqdm
from sqlalchemy import create_engine, inspect
from dotenv import load_dotenv
from utils import checkpoint as cp
import logging
from typing import Union

load_dotenv()
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST", "localhost")
db_port = os.getenv("DB_PORT", "5432")
db_name = os.getenv("DB_NAME")

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
    從 GeoDataFrame 的欄位型別推斷 metadata_schema，（應是用於推斷GIS_001的欄位型別，而不是metadata？）
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

def log_gis_metadata(
    conn,
    area_id: str,
    area_level: str,
    classification: str,
    shp_version: str,
    save_info: dict,
    stage: str = "raw"
) -> bool:
    """
    將單筆影像處理結果寫入 GIS_metadata。
    失敗時 stage 寫入 "failed" 。
    """
    from database_manager import execute_sql

    file_path = save_info.get("path", "")
    # stage 預設由呼叫端的 save_info 路徑推斷，或直接由參數指定
    # 此處統一以參數 stage 為準
    record = {
        "area_id":        area_id,
        "area_level":     area_level,
        "classification": classification,
        "shp_version":    shp_version,
        "stage":          stage,
        "file_path":      file_path,
        "layer_updated":  False
    }
    layer_updated = save_info.get("status") == "updated"
    if layer_updated:
        record["layer_updated"] = True
    upsert_sql = """
        INSERT INTO "GIS_metadata"
            (area_id, area_level, classification, shp_version, stage, file_path, layer_updated)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (area_id, classification, shp_version, stage)
        DO UPDATE SET
            file_path    = EXCLUDED.file_path,
            recorded_at  = NOW();
    """    
    
    result = execute_sql(
        conn, upsert_sql,
        (
            record["area_id"], record["area_level"], record["classification"],
            record["shp_version"], record["stage"], record["file_path"], record["layer_updated"]
        )
    )
    if not result:
        logger.warning(
            f"GIS_metadata 寫入失敗（僅記錄 log，不中止流程）："
            f"area_id={area_id}, classification={classification}, stage={stage}。"
        )
        return False
    logger.debug(
        f"GIS_metadata 寫入成功：area_id={area_id}, "
        f"classification={classification}, stage={stage}。"
    )
    return True

def create_gis_table(conn) -> bool:
    """
    建立 GIS 專案所需的核心資料表：GIS_001 與 GIS_metadata。
    若表格已存在則直接回傳 True。
    """

    # 1. 確認 PostGIS extension 是否安裝
    try:
        postgis_check = execute_sql(
            conn,
            "SELECT 1 FROM pg_extension WHERE extname = 'postgis';",
            fetch_one=True
        )
        if not postgis_check:
            logger.error("❌ PostGIS extension 未安裝！請先執行：CREATE EXTENSION postgis;")
            return False
        logger.notice("✅ PostGIS extension 已安裝")
    except Exception as e:
        logger.error(f"檢查 PostGIS 時發生錯誤: {e}")
        return False
    tables_created = True

    # ====================== 建立 GIS_001 ======================
    if not table_exists(conn, "GIS_001"):
        logger.notice("正在建立資料表 GIS_001...")        
        sql_gis001 = """
        CREATE TABLE IF NOT EXISTS "GIS_001" (
            area_id          TEXT          NOT NULL,
            area_level       TEXT          NOT NULL CHECK (area_level IN ('county', 'town')),
            "COUNTYID"       TEXT,
            "COUNTYCODE"     TEXT,
            "COUNTYNAME"     TEXT,
            "COUNTYENG"      TEXT,
            "TOWNID"         TEXT,
            "TOWNCODE"       TEXT,
            "TOWNNAME"       TEXT,
            "TOWNENG"        TEXT,
            geometry         geometry(Geometry, 4326) NOT NULL,
            shp_version      TEXT          NOT NULL,
            created_at       TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            PRIMARY KEY (area_id),
            UNIQUE (area_id, shp_version)
        );
        -- 僅保留空間索引（GIS 必要），其餘一般索引移除
        CREATE INDEX IF NOT EXISTS idx_gis001_geometry ON "GIS_001" USING GIST (geometry);
        """        
        if execute_sql(conn, sql_gis001):
            logger.success("✅ GIS_001 資料表建立完成")
        else:
            logger.error("❌ GIS_001 建立失敗")
            tables_created = False
    else:
        logger.notice("GIS_001 資料表已存在，跳過建立")

    # ====================== 建立 GIS_metadata ======================
    if not table_exists(conn, "GIS_metadata"):
        logger.notice("正在建立資料表 GIS_metadata...")
        sql_enum = """
        DO $$ BEGIN
            CREATE TYPE gis_stage AS ENUM ('raw', 'masked', 'failed');
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
        """
        execute_sql(conn, sql_enum)
        sql_metadata = """
        CREATE TABLE IF NOT EXISTS "GIS_metadata" (
            id               SERIAL        PRIMARY KEY,
            area_id          TEXT          NOT NULL,
            area_level       TEXT          NOT NULL,
            classification   TEXT          NOT NULL,     -- 圖層英文名稱
            shp_version      TEXT          NOT NULL,
            stage            gis_stage     NOT NULL,
            file_path        TEXT          NOT NULL,
            layer_updated    BOOLEAN       DEFAULT FALSE,
            recorded_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            UNIQUE (area_id, classification, shp_version, stage)
        );
        
        CREATE INDEX IF NOT EXISTS idx_gismeta_batch_load ON "GIS_metadata" (shp_version, classification)
        INCLUDE (stage, area_id, file_path);
        CREATE INDEX IF NOT EXISTS idx_gismeta_file_path ON "GIS_metadata" (file_path);
        """
        if execute_sql(conn, sql_metadata):
            logger.success("✅ GIS_metadata 資料表建立完成")
        else:
            logger.error("❌ GIS_metadata 建立失敗")
            tables_created = False
    else:
        logger.notice("GIS_metadata 資料表已存在，跳過建立")
    if tables_created:
        logger.notice("🎉 create_gis_table 執行完成，所有必要表格已就緒")
        return True
    else:
        logger.error("⚠️ 部分表格建立失敗")
        return False

def load_map_links(conn) -> list[dict]:
    metadata = generate_metadata(conn)

    maps_tableid = [k for k, v in metadata.items() if "102" in k]
    maps_data = []
    for map_tableid in maps_tableid:
        sql_query = f'SELECT * FROM "{map_tableid}";'
        maps_data.extend(
            execute_sql(conn, sql_query, fetch_all=True)
        )
    map_links = {d.get("圖檔中文名稱"): d.get("分布圖Url") for d in maps_data}
    return map_links

def generate_metadata(conn):
    """
    這個函數用來生成 metadata 資料。(字典)
    """
    if not table_exists(conn, "metadata_index"):
        return {}
    sql_query = "SELECT * FROM metadata_index;"
    all_metadata_records = execute_sql(conn, sql_query, fetch_all=True)
    if all_metadata_records:
        metadata_dict = {record['category_table_id']: record for record in all_metadata_records}
        logger.notice(f"INFO: 已生成 {len(metadata_dict)} 條 metadata 記錄。")
        return metadata_dict
    else:
        logger.warning("WARNING: 未能從資料庫生成任何 metadata 記錄。")
        return {}

def load_all_polygon_coords(conn, area_level: str = None) -> gpd.GeoDataFrame:
    """
    從 GIS_001 讀取行政區資料，回傳 GeoDataFrame（geometry 已強制轉為 MultiPolygon）。

    返回值特性：
        - gdf.geometry 全部為 shapely.geometry.MultiPolygon（含單一 Polygon 也自動包裝）
        - 與直接 gpd.read_file(SHP) 的 geometry 格式完全一致
        - 可直接用於後續 BBOX 計算、png_geographic_mapping() 等

    Args:
        conn: moa_gis 連線物件
        area_level: 可選 'county' 或 'town'，未來擴充用

    Returns:
        gpd.GeoDataFrame 或空 DataFrame（失敗時）
    """
    if not table_exists(conn, "GIS_001"):
        logger.warning("GIS_001 表格不存在，無法載入資料。")
        return gpd.GeoDataFrame()

    # 1. 建立 SQLAlchemy Engine（僅此處使用，讀完立即釋放）
    try:
        sa_url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        engine = create_engine(sa_url, echo=False)
        logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    except Exception as e:
        logger.error(f"建立 SQLAlchemy engine 失敗: {e}")
        return gpd.GeoDataFrame()

    # 2. SQL 強制轉 MultiPolygon
    where_clause = ""
    params = None
    if area_level in ("county", "town"):
        where_clause = "WHERE area_level = %s"
        params = (area_level,)
    try:
        # 1. 取得表格的所有欄位名稱
        inspector = inspect(engine)
        all_columns = [col['name'] for col in inspector.get_columns("GIS_001")]
        
        # 2. 過濾掉原始的 geometry 欄位
        # 使用列表推導式，確保不區分大小寫
        other_columns = [f'"{c}"' for c in all_columns if c.lower() != 'geometry']
        cols_str = ", ".join(other_columns)

        # 3. 組合成新的 SQL
        query = f"""
            SELECT 
                {cols_str}, 
                ST_Multi(geometry) AS geometry
            FROM "GIS_001"
            {where_clause}
            ORDER BY area_id;
        """
        gdf = gpd.read_postgis(
            sql=query,
            con=engine,
            geom_col="geometry",
            params=params,
            crs="EPSG:4326"
        )
    except Exception as e:
        logger.error(f"read_postgis 失敗: {e}")
        return gpd.GeoDataFrame()
    finally:
        engine.dispose()

    if gdf.empty:
        logger.warning(f"GIS_001 查無資料（area_level={area_level}）")
        return gdf

    logger.info(f"已成功載入 {len(gdf)} 筆行政區資料（全部為 MultiPolygon）")
    return gdf

def get_gis_metadata(conn, para, shp_version: str, area_level: str = None, area_id: str = None, classification: str = None) -> pd.DataFrame:
    """
    para: 未來擴充用參數
    """
    pass

def check_gis_exists(conn, save_info: dict, area_id: str, map_name_en: str, shp_version: str, stage: str = "masked") -> Union[bool, str]:
    if save_info["status"] != "unchanged":
        return False  # 只有 unchanged 才需要檢查

    check_sql = """
        SELECT file_path FROM "GIS_metadata"
        WHERE area_id = %s
          AND classification = %s
          AND shp_version = %s
          AND stage = %s
        LIMIT 1;
    """
    meta_result = execute_sql(
        conn, check_sql,
        (area_id, map_name_en, shp_version, stage),
        fetch_one=True
    )
    logger.debug(f"meta_result: {meta_result}")
    if meta_result and os.path.exists(meta_result[0]):
        return meta_result[0]

    return False

def check_shp_needs_update(conn, shp_version: str) -> bool:
    """
    判斷 shp_version 是否需要執行入庫。
    基於「失敗即回滾」策略，只需確認是否有任何一筆紀錄存在。
    """
    # 這裡直接執行 EXISTS 檢查，這是最快路徑
    # 只要 shp_version 索引存在，這幾乎不消耗時間
    sql = 'SELECT 1 FROM "GIS_001" WHERE shp_version = %s LIMIT 1;'
    
    try:
        # fetch_one=True 會回傳第一筆資料，若無資料則回傳 None
        result = execute_sql(conn, sql, (shp_version,), fetch_one=True)
        
        if result:
            return False  # 已存在，不需要更新
        
        logger.notice(f"🚀 版本 {shp_version} 尚未入庫，準備開始作業...")
        return True       # 需要更新
        
    except Exception as e:
        logger.error(f"檢查版本時出錯: {e}")
        # 出錯時保險起見回傳 False，避免在連線異常時意外啟動入庫程序
        return False

def upsert_gis_boundary(conn, gdf: gpd.GeoDataFrame, shp_version: str) -> bool:
    """
    將 shp_reader() 讀取的 GeoDataFrame 批次 upsert 至 GIS_001 資料表。
    """
    total = len(gdf)

    # 1. 判定層級與衝突欄位
    # 修正：同時考慮 TOWNID 是否在 columns 中以及其內容是否為空
    is_county_level = 'TOWNID' not in gdf.columns
    area_level = "county" if is_county_level else "town"
    
    # 複合衝突鍵：確保同一個區域在同一個版本中不會重複
    conflict_target = '"area_id", "shp_version"'
    logger.notice(f"處理【{area_level}】層級資料，衝突鍵：{conflict_target}，版本：{shp_version}")

    # 2. 確保欄位（排除幾何，並包含我們手動補入的 area_id 等）
    # 這裡 ensure_columns_exist 會自動 ALTER TABLE 補齊原本 gdf 沒有但 SQL 需要的欄位
    columns_to_ensure = [col for col in gdf.columns if col.lower() != 'geometry']
    for extra_col in ['area_id', 'area_level', 'shp_version']:
        if extra_col not in columns_to_ensure:
            columns_to_ensure.append(extra_col)
            
    if not ensure_columns_exist(conn, "GIS_001", columns_to_ensure):
        return False

    # 3. 幾何合法性修正 (Topological cleaning)
    logger.info(f"執行幾何合法性檢查：make_valid + buffer(0)...")
    # 未來若要處理「村里」級別，建議改在迴圈內逐筆處理 record['geometry'].make_valid()，或是確認 SHP 讀取時就已經是乾淨的
    gdf['geometry'] = gdf['geometry'].make_valid().buffer(0)

    # 4. 【關鍵】參數與 SQL 組裝
    # 基礎欄位：排除幾何、版本以及我們會手動處理的 area_ 欄位
    base_cols = [c for c in gdf.columns if c.lower() not in ['geometry', 'shp_version', 'area_id', 'area_level']]
    
    # 組裝 SQL 欄位順序：area_id, area_level, [gdf 欄位], geometry, shp_version
    all_cols_sql = ['"area_id"', '"area_level"'] + \
                   [f'"{c}"' for c in base_cols] + \
                   ['"geometry"', '"shp_version"']
    
    placeholders = "%s, %s, " + \
                   ", ".join(["%s"] * len(base_cols)) + \
                   ", ST_GeomFromText(%s, 4326), %s"
    
    upsert_sql = f"""
        INSERT INTO "GIS_001" ({", ".join(all_cols_sql)}) 
        VALUES ({placeholders})
        ON CONFLICT ({conflict_target}) 
        DO NOTHING; 
    """

    # 5. 批次執行
    logger.notice(f"開始批次寫入行政區界到GIS_001，共 {total} 筆（多版本模式）")
    data_records = gdf.to_dict('records')
    ver_town = "TOWN_MOI"
    
    try:
        for record in tqdm(data_records, desc=f"寫入行政區界", unit="row"):
            # 產生 area_id
            area_id = record.get('COUNTYID') if is_county_level else record.get('TOWNID')
            if area_id == "T28" and ver_town.lower() in shp_version.lower():
                total -= 1
                town_name = record.get("TOWNNAME")
                logger.info(f"從鄉鎮列表中移除 {town_name}，由Town_Majia_Sanhe補入。")
                continue
            
            # === 資料完整性強制檢查 ===
            if not area_id or str(area_id).strip() == "":
                conn.rollback()
                logger.critical(f"【嚴重錯誤】發現缺少 area_id 的資料列！ "
                                f"layer={area_level}, shp_version={shp_version}")
                logger.error(f"問題資料內容: {record}")
                return False
            
            # 組裝參數
            params = [
                area_id,                    # area_id
                area_level,                 # area_level
            ]
            params.extend([record.get(c) for c in base_cols])   # 其他屬性欄位
            params.append(record['geometry'].wkt)               # geometry
            params.append(shp_version)                          # shp_version
            
            # 執行寫入
            success = execute_sql(conn, upsert_sql, tuple(params), auto_commit=False)
            
            if not success:
                conn.rollback()
                error_id = area_id
                logger.error(f"寫入失敗，area_id: {error_id}，已全域回滾")
                return False

        # 全部成功才 commit
        conn.commit()
        logger.success(f"行政區界入庫完成！版本 {shp_version}，共 {total} 筆資料。")
        return True

    except Exception as e:
        conn.rollback()
        logger.error(f"upsert_gis_polygon 發生未預期錯誤: {e}，已回滾")
        return False

def clear_failed_records(conn, shp_ver: str) -> None:
    """
    清除該 shp_ver (純數字) 下 stage 標記為 failed 的紀錄，要在 get_missing_records 回傳為空時使用。
    """
    count_sql = """
        SELECT COUNT(*) FROM "GIS_metadata"
        WHERE shp_version LIKE %s
          AND stage = 'failed';
    """
    count_result = execute_sql(conn, count_sql, (f"%{shp_ver}%",), fetch_one=True)
    count = count_result[0] if count_result else 0

    if count == 0:
        logger.info(f"版本 {shp_ver} 無失敗紀錄，略過清除。")
        return

    delete_sql = """
        DELETE FROM "GIS_metadata"
        WHERE shp_version LIKE %s
          AND stage = 'failed';
    """
    execute_sql(conn, delete_sql, (f"%{shp_ver}%",))
    logger.info(f"已清除 {count} 筆版本 {shp_ver} 的失敗紀錄。")

def get_missing_records(conn, shp_ver: str, all_area_ids: list, all_classifications: list) -> list[dict]:
    """
    找出該 shp_ver (純數字) 下「應該處理但 GIS_metadata 中完全沒有紀錄」的組合
    """
    if not all_area_ids or not all_classifications:
        return []

    sql = """
        SELECT DISTINCT area_id, classification 
        FROM "GIS_metadata" 
        WHERE shp_version LIKE %s
          AND stage = 'masked';
    """
    existing = execute_sql(conn, sql, (f"%{shp_ver}%",), fetch_all=True) or []
    existing_set = {(row['area_id'], row['classification']) for row in existing}

    missing = []
    for area_id in all_area_ids:
        for cls in all_classifications:
            if (area_id, cls) not in existing_set:
                missing.append({
                    "area_id": area_id,
                    "classification": cls
                })

    return missing

if __name__ == '__main__':

    # 參考
    # 一次撈取該版本的所有 Metadata
    sql = """
        SELECT area_id, classification, stage, file_path 
        FROM "GIS_metadata" 
        WHERE shp_version = %s
    """
    raw_data = execute_sql(conn, sql, (current_version,), fetch_all=True)

    # 建立分類「抽屜」：複合鍵映射表
    # Key: (行政區, 類別, 狀態) -> Value: 路徑
    metadata_lookup = {
        (r['area_id'], r['classification'], r['stage']): r['file_path'] 
        for r in raw_data
    }

    # 使用方式：
    # path = metadata_lookup.get(('A01', 'land_use', 'masked'))

    import os

    def get_disk_files(root_dir):
        # 建立 Set 加快 in 比對速度 (O(1))
        # 存儲檔名或相對路徑
        return {f.name for f in os.scandir(root_dir) if f.is_file()}

    # disk_set = get_disk_files("./data/v2026/land_use")