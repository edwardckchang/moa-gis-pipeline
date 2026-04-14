import psycopg2
from logs_handle import logger
from typing import List, Dict, Optional, Any

def connect_conn(username, password, dbname, host="localhost", port=5432):
    try:
        conn_str = f"host={host} port={port} dbname={dbname} user={username} password={password}"
        conn = psycopg2.connect(conn_str)
        if conn:
            logger.notice(f"成功連線到PostgreSQL資料庫: {dbname}@{host}:{port}")
    except Exception as e:
        logger.error(f"無法連接到資料庫 {dbname}@{host}:{port}: {e}")
        conn = None
    return conn

def disconnect_conn(conn):
    if conn:
        conn.close()
        logger.notice("資料庫連線已關閉。")

def _ensure_connection(conn: Optional[psycopg2.extensions.connection]) -> bool:
    """
    確認 connection 物件存在且連線未關閉。

    Args:
        conn: 待確認的 connection 物件。

    Returns:
        可用則回傳 True；否則記錄錯誤並回傳 False。
    """
    if conn is None or conn.closed:
        logger.error("資料庫未連線或連線已關閉。")
        return False
    return True

def execute_sql(conn, sql_query: str, params: tuple = None, fetch_one: bool = False, fetch_all: bool = False):
    """
    執行 SQL 查詢，並可選擇返回單條或所有結果。
    自動處理游標關閉和事務提交/回滾。
    """
    if not _ensure_connection(conn):
        return None

    cur = conn.cursor()
    try:
        logger.debug(f"執行 SQL: {sql_query}\nParams: {params}")
        cur.execute(sql_query, params)
        
        result = None
        if fetch_one:
            result = cur.fetchone()
        elif fetch_all:
            columns = [desc[0] for desc in cur.description]
            result = [dict(zip(columns, row)) for row in cur.fetchall()]
        else:
            # 對於非查詢操作（如不帶 RETURNING 的 INSERT/UPDATE/DELETE），返回 True 表示成功
            result = True 
        
        # 關鍵修改：在執行成功後提交事務
        # 這會確保所有 DML 操作（包括帶 RETURNING 的）都能持久化
        conn.commit() 
        
        return result

    except Exception as e:
        conn.rollback() # 發生錯誤時回滾事務
        logger.error(f"執行 SQL 失敗: {type(e).__name__}: {e}\nSQL: {sql_query}\nParams: {params}")
        return False # 查詢或操作失敗
    finally:
        cur.close()

def generate_global_data():
    """
    這個函數用來生成 GLOBAL_METADATA_CACHE 的值。(字典)
    """
    if not _ensure_connection(conn):
        return {}
    if not table_exists("metadata_index"):
        return {}
    logger.notice("INFO: 正在從資料庫生成 metadata...")
    sql_query = "SELECT * FROM metadata_index;"
    all_metadata_records = execute_sql(sql_query, fetch_all=True)
    if all_metadata_records:
        for record in all_metadata_records:
            table_name = record.get('category_table_id')
            if table_name and table_exists(table_name):
                count_sql = f"SELECT COUNT(*) as total_count FROM \"{table_name}\";"
                count_res = execute_sql(count_sql, fetch_one=True)
                record['資料筆數'] = count_res[0] if count_res else 0
            else:
                record['資料筆數'] = 0
        metadata_dict = {record['category_table_id']: record for record in all_metadata_records}
        logger.notice(f"INFO: 已生成 {len(metadata_dict)} 條 metadata 記錄。")
        return metadata_dict
    else:
        logger.warning("WARNING: 未能從資料庫生成任何 metadata 記錄。")
        return {}
        
def table_exists(conn, table_name: str) -> bool:
    """
    檢查指定表格是否存在於資料庫中。
    """
    if not _ensure_connection(conn):
        return False
    cur = conn.cursor()
    try:
        table_exists_sql = f"""
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = '{table_name}'
        );
        """
        cur.execute(table_exists_sql)
        return cur.fetchone()[0]
    except Exception as e:
        logger.error(f"檢查表格 '{table_name}' 是否存在失敗: {e}")
        return False
    finally:
        cur.close()
        
def get_all_tables(conn) -> List[str]:
    """
    獲取資料庫中所有表格的名稱。
    """
    if not _ensure_connection(conn):
        return []
    cur = conn.cursor()
    try:
        sql_query = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public';
        """
        cur.execute(sql_query)
        # fetchall() returns a list of tuples, get the first element of each tuple
        tables = [row[0] for row in cur.fetchall()]
        return tables
    except Exception as e:
        logger.error(f"獲取所有表格名稱失敗: {e}")
        return []
    finally:
        cur.close()
        
def ensure_columns_exist(conn, table_name: str, columns: list[str]) -> bool:
    """
    確保表格中存在指定的所有欄位，若缺少則自動新增（類型預設 TEXT）。
    columns 為欄位名稱字串列表（未加引號的原始名稱）。
    """
    if not _ensure_connection(conn):
        return False
    existing_result = execute_sql(
        "SELECT column_name FROM information_schema.columns WHERE table_schema = 'public' AND table_name = %s;",
        (table_name,), fetch_all=True
    )
    if existing_result is False or existing_result is None:
        logger.error(f"ensure_columns_exist: 無法取得表格 '{table_name}' 的欄位資訊。")
        return False
    existing_cols = {row['column_name'] for row in existing_result}
    for col in columns:
        if col not in existing_cols:
            alter_sql = f'ALTER TABLE "{table_name}" ADD COLUMN "{col}" TEXT;'
            if execute_sql(alter_sql):
                logger.warning(f"表格 '{table_name}' 新增欄位: {col}")
            else:
                logger.error(f"表格 '{table_name}' 新增欄位 '{col}' 失敗。")
                return False
    return True

GIS_TYPE_MAP = {
    "text":     "TEXT",
    "integer":  "BIGINT",
    "float":    "DOUBLE PRECISION",
    "boolean":  "BOOLEAN",
    "datetime": "TIMESTAMP WITH TIME ZONE",
    "json":     "JSONB",
    "geometry": "geometry(Geometry, 4326)",  # PostGIS
}

def table_columns_sql(metadata_schema: dict) -> str:
    """
    將 metadata_schema 轉換為 PostgreSQL 欄位定義 SQL 字串。
    metadata_schema 的 value 為 GIS_TYPE_MAP 中定義的字串標籤。
    """
    col_definitions = []
    for col_name, col_type in metadata_schema.items():
        if col_name == "foreign_key":
            continue
        pg_type = GIS_TYPE_MAP.get(col_type, "TEXT")
        if pg_type == "TEXT" and col_type not in GIS_TYPE_MAP:
            logger.warning(f"未知的欄位型別標籤 '{col_type}'（欄位：'{col_name}'），退回 TEXT。")
        safe_col_name = f'"{col_name.replace(" ", "_").replace(".", "_").replace("-", "_")}"'
        col_definitions.append(f'{safe_col_name} {pg_type}')
    return ", ".join(col_definitions)

def execute_upsert(
    conn: psycopg2.extensions.connection,
    table_name: str,
    record: Dict[str, Any],
    conflict_columns: List[str],
) -> bool:
    """
    對單一筆記錄執行 INSERT … ON CONFLICT DO UPDATE（upsert）。

    欄位名稱以雙引號包裹，支援中文欄位名。
    衝突判斷依 conflict_columns；更新範圍為 record 中所有非衝突欄位。
    若所有欄位都是衝突欄位（即無需更新的欄位），則改為 DO NOTHING。

    Args:
        conn:             使用中的 connection 物件。
        table_name:       目標資料表名稱。
        record:           欲寫入的資料（dict，key 為欄位名）。
        conflict_columns: 用於 ON CONFLICT 判斷的欄位名稱列表。

    Returns:
        成功回傳 True；失敗回傳 False。
    """
    if not _ensure_connection(conn):
        return False
    if not record:
        logger.warning("execute_upsert：record 為空，略過。")
        return False

    cols = list(record.keys())
    quoted_cols = [f'"{c}"' for c in cols]
    placeholders = ", ".join(["%s"] * len(cols))
    conflict_target = ", ".join([f'"{c}"' for c in conflict_columns])

    update_cols = [c for c in cols if c not in conflict_columns]
    if update_cols:
        update_clause = ", ".join(
            [f'"{c}" = EXCLUDED."{c}"' for c in update_cols]
        )
        on_conflict = f"ON CONFLICT ({conflict_target}) DO UPDATE SET {update_clause}"
    else:
        on_conflict = f"ON CONFLICT ({conflict_target}) DO NOTHING"

    sql = (
        f'INSERT INTO "{table_name}" ({", ".join(quoted_cols)}) '
        f"VALUES ({placeholders}) "
        f"{on_conflict};"
    )
    values = tuple(record[c] for c in cols)

    result = execute_sql(conn, sql, values)
    if result is False:
        logger.error(
            f"execute_upsert 失敗：table={table_name}, "
            f"conflict_columns={conflict_columns}"
        )
        return False
    return True

if __name__ == "__main__":
    from dotenv import dotenv_values
    config = dotenv_values() # 讀取 .env 檔案中的值
    DB_USER = config.get("DB_USER")
    DB_PASSWORD = config.get("DB_PASSWORD")
    DB_NAME = config.get("DB_NAME")
    conn = connect_conn(DB_USER, DB_PASSWORD, DB_NAME)
    if not conn:
        print("連線失敗，程式結束。")
    
    