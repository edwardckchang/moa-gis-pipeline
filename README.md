---

# 農業開放資料平台資料庫工具

從農業部開放資料平台（data.moa.gov.tw）自動下載 JSON 資料集，並存入本地 PostgreSQL 資料庫的 CLI 工具。支援排程自動更新、資料集管理與維護操作。

---

## 系統需求

- Python 3.9+
- PostgreSQL（本地，預設 `localhost:5432`）
- 相依套件：`psycopg2`、`pandas`、`requests`、`beautifulsoup4`、`python-dotenv`、`tqdm`

---

## 環境設定

在專案根目錄建立 `.env`：

```
USERNAME=your_db_user
PASSWORD=your_db_password
DBNAME=your_db_name
```

---

## 入口點

| 腳本 | 用途 |
|---|---|
| `main.py` | 主選單，手動下載 / 更新 / 資料庫管理 |
| `auto_update_data_moa.py` | 自動批次更新所有已記錄資料集 |
| `operations_of_postgresql.py` | 獨立執行資料庫管理選單 |

---

## 主選單（`main.py`）

```
請選擇操作：
1. 輸入目標網址下載並儲存資料
2. 更新統計資料
3. 待審核資料處理（未實作）
4. 進入資料庫操作系統
q. 退出
```

### 選單 1：手動下載

次選單提供三種方式：直接輸入介紹頁網址、從 metadata 清單選擇、或重新下載所有資料集（`fully_auto_update` + `autodetect_skip=True`）。

核心流程：`_handle_data_download(webpage_url, user_inputs)` → `parse_webpage_to_metadata()` → `fetch_and_process_json_data()` → 存入 PostgreSQL

### 選單 2：更新統計資料（`update_by_metadata`）

從 `minor_info.refer_skip_value` 取得所有已記錄資料集，提供自動全部更新（`fully_auto_update`）或從清單選擇單筆更新。

### 全自動更新（`fully_auto_update`）

逐筆執行 `_handle_data_download`，每筆間隔 1 秒，結束後寫回 `minor_info`（JSON + SQL）並輸出執行結果摘要。

---

## 資料庫管理選單（`operations_of_postgresql.py`）

從 `main.py` 選單 4 進入，或直接執行 `operations_of_postgresql.py`：

```
--- PostgreSQL 資料庫操作 ---
0. 連線資料庫
1. 顯示所有資料集清單（含 ID 與標題）
2. 查詢並顯示特定資料集簡介
3. 重新命名資料集
4. 刪除資料集
5. 匹配資料表索引與表格名稱
6. 刪除資料集內重複資料
7. 建立資料集索引（加速作業）
q. 退出
```

| 選項 | 對應函式 | 說明 |
|---|---|---|
| 1 | `_get_and_display_metadata_list()` | 列出所有資料集概要 |
| 2 | `_search_and_display_datasets()` | 關鍵字或 ID 搜尋，支援精確 / 模糊匹配 |
| 3 | `_rename_dataset_table()` | 重新命名資料表與 metadata 記錄 |
| 4 | `_delete_dataset_option()` | 次選單：依清單 / 分類 / 關鍵字刪除，或僅刪除 processed 表格 |
| 5 | `_match_table_indexes_and_names()` | 同步 metadata_index 與資料庫實際表格清單 |
| 6 | `_delete_replicate_data()` | 依 `all_sort_configs` 鍵值，以 `ROW_NUMBER() OVER (PARTITION BY ...)` 識別並刪除重複列 |
| 7 | `create_indexes_for_all_tables()` | 對所有資料表建立索引（來自 `db_maintenance.py`） |

---

## 模組結構

```
main.py                      主入口、主選單、下載/更新流程
auto_update_data_moa.py      自動排程更新入口
operations_of_postgresql.py  資料庫管理 CLI 入口與所有 _ 前綴操作函式
database_manager.py          DB 連線、SQL 執行、Repository 層、Cache 層
data_parser.py               網頁解析、JSON 下載、資料前處理
db_maintenance.py            索引建立等維護作業
json_file_operations.py      metadata.json / minor_info 的讀寫操作
sort_data_by_date.py         資料排序設定載入
logs_handle.py               自訂 logging（含 notice / success / execution 等級）
menu_utils.py                yes_no_menu、AUTO_YES/NO 等互動輔助
utils.py                     通用工具函式（clean_table_name、select_row_by_index 等）
ui_interface.py              Tkinter 彈窗輸入介面
```

---

## 資料模型

**`metadata_index`**：各資料集 metadata，主鍵 `category_table_id`（分類碼 3 位 + 序列 4 位，例如 `1040012`）

**`<category_table_id>`**：原始資料表，主鍵 `category_table_data_id`（`1` + id）

**`<category_table_id>_processed`**：處理後資料表，FK 參照原始表（`2` + id）

**`files`**：檔案記錄表（PDF、圖檔等）

**`record_files`**：資料記錄與檔案的橋接表

**`refer_skip_value`** / **`all_sort_configs`**：`minor_info` 的 SQL 映射表

---

## 狀態初始化流程（`main.py __main__`）

1. 載入 `.env`，取得 `USERNAME / PASSWORD / DBNAME`
2. `connect_db()` 建立連線
3. `init(DB)` 初始化 `global_metadata_cache`（`get_global_data()`）與 `minor_info`（`get_minor_info_data()`）
4. 確保 `metadata_index`、`files`、`record_files` 表格存在
5. 進入 `main()` 選單迴圈

---