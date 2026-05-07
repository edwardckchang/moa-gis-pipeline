# moa-gis-pipeline Issues

## 狀態說明
- ⬜ 未開始
- 🔄 進行中
- ✅ 已完成

---

## Phase 1：專案基礎建設

### Issue 1：專案初始化與骨架建立
**狀態：** ✅ 已完成

#### 開發環境
- [X] 在新專案根目錄執行 `git init`
- [X] 建立 `.gitignore`
  - 排除：`.env`、`output/*`、`logs/*`、`__pycache__/`、`*.pyc`、`*.parquet`、`temp.*`、
          `test.*`、`metadata/`、`old_file/`、`farmland_spatial_map/*.json`
  - 保留：`output/.gitkeep`、`logs/.gitkeep`
- [X] 建立 `.env`（含 `DB_USER`、`DB_PASSWORD`、`DB_NAME`）
- [X] 建立 `.env.example`（含欄位名稱但值為空，提交進 git）
- [X] 建立 `requirements.txt`
  - 套件：`geopandas`、`rasterio`、`shapely`、`psycopg2`、`pandas`、`numpy`、`Pillow`、
          `vincenty`、`tqdm`、`python-dotenv`、`locale`、`json`
  - 使用pip-chill > requirements.txt，每個階段完成須執行一次
- [X] 確認 Python 3.9 環境可正常安裝所有套件

#### 資料夾結構
- [X] 建立 `GIS/`（放置 SHP 檔案用）
- [X] 建立 `output/`（加入 `.gitkeep` 佔位）
- [X] 建立 `logs/`（加入 `.gitkeep` 佔位）

#### 繼承檔案處理
- [X] 複製 `logs_handle.py`，不修改
- [X] 複製 `geographic_color_metadata.json`，不修改，確認進入 git 追蹤

#### `utils.py` 重構
- [X] 複製 `utils.py`
- [X] 從 `build_base_para.py` 遷移：`process_string()`
- [X] 從 `operations_of_postgresql.py` 遷移：`_clean_string()`

#### `sort_utils.py` 建立
- [X] 從舊 `sort_data_by_date.py` 遷移：`sort_list_of_dictionaries()`
- [X] 從舊 `sort_data_by_date.py` 遷移：`_interactive_sort_data()`
- [X] 確認兩函式的 import 依賴只有 `utils.py` 和 `logs_handle.py`，無舊專案耦合
- [X] 確認額外需要 import：`tqdm`、`locale`、`json`（均為標準庫或 requirements.txt 已列套件）

#### `file_utils.py` 建立
- [X] 從舊 `json_file_operations.py` 遷移：`load_json_data()`
- [X] 從舊 `json_file_operations.py` 遷移：`save_json_data()`

#### `cli_utils.py` 建立
- [X] 複製並改名 `menu_utils.py` 為：`cli_utils.py`
- [X] `handle_save_menu()`修正 bug：`question` 變數在 `if name:` 條件外被使用，需補上 else 分支或移至條件外

#### `database_manager.py` 重構
- [X] 架構改動：移除模組層級全域 `DB` 變數，改為 `conn` 物件由呼叫端持有
  - 原因：避免模組間隱式狀態共享，conn 生命週期由 main_gis.py 統一管理
- [X] 從舊版遷移並修改簽名：`connect_db()`（每次呼叫回傳新 conn，不修改全域狀態）
- [X] 新增：`disconnect_db(conn)`
- [X] 從舊版遷移並修改簽名：`_ensure_connection(conn)`（原名 `_ensure_db_connection`）
- [X] 從舊版遷移並修改簽名：`execute_sql(conn, sql, params, *, fetch_one, fetch_all)`
- [X] 從舊版遷移並修改簽名：`table_exists(conn, table_name)`
- [X] 從舊版遷移並修改簽名：`get_all_tables(conn)`
- [X] 從舊版遷移並修改簽名：`ensure_columns_exist(conn, table_name, columns)`
- [X] 從舊版遷移：`table_columns_sql()`
- [X] 新增：`execute_upsert(conn, table_name, record, conflict_columns)`
- [X] 以下全部不遷移：
  - `Application_Area`
  - `GLOBAL_METADATA_CACHE` 及相關函式
  - `metadata_index` 相關所有函式
  - `minor_info` 相關所有函式
  - `files` / `record_files` 表相關所有函式
  - `preserve_old_data_by_date()`
  - `get_global_data()` / `get_minor_info_data()`

#### 空白骨架模組建立
- [X] 建立 `gis_db.py`（含 docstring 說明用途與 TODO 標注）
- [X] 建立 `gis_reader.py`（含 docstring 與 TODO 標注）
- [X] 建立 `gis_downloader.py`（含 docstring 與 TODO 標注）
- [X] 建立 `gis_quantifier.py`（含 docstring 與 TODO 標注）
- [X] 建立 `analysis_pipeline.py`（含 docstring 與 TODO 標注）
- [X] 建立 `main_gis.py`（含 docstring 與 TODO 標注）
- [X] 建立 `main_pipeline.py`（含 docstring，標注 Phase 5 實作）

#### 驗證
- [X] 所有模組可正常 import 不報錯
- [X] `database_manager.py` 可成功連線至本機 PostgreSQL
- [X] `git status` 確認 `.env` 未被追蹤，其餘檔案正常
- [X] 進行第一次 `git commit`

---

### Issue 2：函式預填入骨架模組
**狀態：** ✅ 已完成
**前置條件：** Issue 1 完成

#### `gis_reader.py` 填入
- [X] 從舊 `GIS_reader.py` 遷移：`natural_sort_key()`
- [X] 從舊 `GIS_reader.py` 遷移：`get_width_height_from_geographic_mapping()`
  - 移除 `print()` 改用 `logger.notice()`
- [X] 從舊 `GIS_reader.py` 遷移：`png_geographic_mapping()`
  - 介面調整：接受 `bytes` 物件而非 `png_path` 字串
  - 原因：對話三 WMS 下載後不落地，直接在記憶體處理
- [X] 從舊 `GIS_reader.py` 遷移：`shp_reader()`
- [X] 確認所有 `print()` 改為 `logger` 對應層級

#### `gis_downloader.py` 填入
- [X] 從舊 `GIS_reader.py` 遷移：`replace_url_parameters()`
  - 注意：`data_parser.py` 有完全相同的函式，取一份即可
- [X] 從舊 `data_parser.py` 遷移：`_download_and_save_image()`，由`fetch_wms_image()`取代
- [X] 新增：`save_mask_image(png, classification, country_name)`
  -從`gis_quantifier.py` 遷移

#### `gis_quantifier.py` 填入
- [X] 從舊 `build_base_para.py` 遷移：`load_image_with_chinese_path()`
- [X] 從舊 `build_base_para.py` 遷移：`decode_png_color_value()`
  - 移除 `print()` 改用 `logger`

#### `gis_db.py` 填入
- [X] 新增函式`infer_schema_from_geodataframe(gdf: gpd.GeoDataFrame)`
- [X] 從舊 `main.py` 遷移：`get_count(conn, table_name)`（加 `conn` 參數）
- [X] 從舊 `build_base_para.py` 遷移：`add_color_mapping_level1()`
  - 修正 bug：第 123 行 `png.dtype` 參照未定義變數，改為固定 `np.uint8`
  - 調整：`METADATA_FILE` 路徑改為新專案的 `geographic_color_metadata.json`

#### `analysis_pipeline.py` 填入
- [X] 從舊 `data_preprocessor.py` 遷移：`analyze_and_clean_dataframe()`

#### import 路徑統一檢查
- [X] 所有遷移的函式確認 import 來源已改為新專案模組
  - 應使用：`logs_handle`、`utils`、`file_utils`、`cli_utils`、`database_manager`
- [X] 確認沒有任何模組仍 import 舊專案的
  - `menu_utils`、`json_file_operations`、`sort_data_by_date`、`database_manager`（舊版）

#### 驗證
- [X] 逐一 import 各模組確認無錯誤
- [X] `gis_reader.natural_sort_key()` 單元測試
  - 輸入 `["TOWN10", "TOWN2", "TOWN1"]`，排序結果應為 `["TOWN1", "TOWN2", "TOWN10"]`
- [X] `gis_reader.get_width_height_from_geographic_mapping()` 單元測試
  - 輸入台灣全島 BBOX `(119.3, 21.9, 122.1, 25.4)`，結果應約為 `(311, 389)`（100m/px），實際：`(2857, 3876)`
- [x] `git commit`

---

### Issue 3：版本控制與資料備份策略建立
**狀態：** ✅ 已完成
**前置條件：** Issue 1 完成

#### Git 使用規範建立
- [X] 確認 `.gitignore` 涵蓋所有不應追蹤的檔案
- [X] 在 GitHub 建立 public repository
- [X] 推送 Issue 1、Issue 2 的成果作為初始 commit
- [X] 建立 commit message 規範：
  - `[SETUP]` 環境與骨架
  - `[REFACTOR]` 重構與遷移
  - `[FEATURE]` 新功能
  - `[FIX]` 錯誤修正
  - `[DATA]` 資料相關變動
  - `[DOCS]` 文件、備忘錄、分析筆記

#### 資料庫備份策略
- [X] 確認 `pg_dump` 可正常執行
  - 備份指令：
    $env:PGPASSWORD = "你的資料庫密碼"
    $date = Get-Date -Format "yyyyMMdd"
    pg_dump -U $DB_USER -d $DB_NAME -f "backups/moa_opendata/backup_$($date).sql"
    新增檔案 `backup_db.py` 與函式 `run_manual_backup` 用於手動備份。
- [X] 建立備份目錄結構（加入 `.gitignore`）
  ```
  backups/
  └── moa_opendata/    ← GIS 資料與農業統計數據共用同一資料庫，統一備份
  ```
- [X] 決定備份頻率（建議：每個 Phase 完成後手動執行一次）
- [X] 確認備份檔案會同步到外部儲存（USB）

#### `geographic_color_metadata.json` 定位確認
- [X] 確認此檔案進入 git 追蹤（設定檔性質，幾 KB，適合 git 管理）
- [X] 確認此檔案**不入庫**
  - 原因：`color_to_value` 為異質結構，入庫複雜度高、收益低；git 已提供版本追蹤能力
- [X] 在 `gis_db.py` 的 docstring 中記錄此決策

#### 輸出檔案備份策略
- [X] 確認 `output/` 目錄下的 CSV/Parquet 不進入 git
  ```
- [X] 確認輸出檔案會定期備份到外部儲存

#### 驗證
- [X] GitHub repository 可正常推送與拉取
- [X] `pg_dump` 執行成功，備份檔案可正常還原（`psql` 測試）
- [X] `git log` 確認有清晰的 commit 歷史

---

## Phase 2：地理界線解析與入庫

### Issue 4：SHP 讀取與地理資料入庫
**狀態：** ✅ 已完成
**前置條件：** Phase 1 全部完成

#### `gis_db.py` — 資料表建立

- [X] 定義資料表名稱常數：
  ```python
  TABLE_GIS_001 = "GIS_001"，直接填入字串於程式中
  TABLE_GIS_METADATA = "GIS_metadata"，直接填入字串於程式中
  ```
- [X] 實作 `create_gis_table(conn) -> bool`：
  - [X] 確認 PostGIS extension 已安裝：`SELECT 1 FROM pg_extension WHERE extname = 'postgis';`
  - [X] 若未安裝則 `logger.error` 並回傳 `False`
  - [X] 若兩張表皆已存在，直接回傳 `True`
  - [X] 建立 `GIS_001` 資料表，同時儲存縣市界（COUNTY）與鄉鎮市區界（TOWN），schema：
    - `area_id TEXT PRIMARY KEY`（唯一識別碼，COUNTY 層級填 `COUNTYID`，TOWN 層級填 `TOWNID`）
    - `area_level TEXT`（`"county"` 或 `"town"`）
    - `"COUNTYID" TEXT`
    - `"COUNTYCODE" TEXT`
    - `"COUNTYNAME" TEXT`
    - `"COUNTYENG" TEXT`
    - `"TOWNID" TEXT`（COUNTY 層級填 `NULL`）
    - `"TOWNCODE" TEXT`（COUNTY 層級填 `NULL`）
    - `"TOWNNAME" TEXT`（COUNTY 層級填 `NULL`）
    - `"TOWNENG" TEXT`（COUNTY 層級填 `NULL`）
    - `geometry geometry(Geometry, 4326)`
    - `shp_version TEXT`
    - `created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()`
  - [X] 建立 `GIS_metadata` 資料表，schema：
    - `id SERIAL PRIMARY KEY`
    - `area_id TEXT`（對應 `GIS_001.area_id`，COUNTY 層級為 `COUNTYID`，TOWN 層級為 `TOWNID`）
    - `area_level TEXT`（`"county"` 或 `"town"`）
    - `classification TEXT`（圖層英文名稱）
    - `shp_version TEXT`
    - `stage TEXT`（`"raw"` 或 `"masked"`）
    - `file_path TEXT`
    - `recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()`
    - UNIQUE constraint on `(area_id, classification, shp_version, stage)`
  - [X] 呼叫 `execute_sql()` 執行兩段 CREATE TABLE SQL
  - [X] 再次以 `table_exists()` 確認兩張表建立成功後回傳 `True`
- [X] 實作 `check_shp_needs_update(conn, shp_version: str) -> bool`：精簡化
  - [X] 查詢 `GIS_001` 確認此 `shp_version` 是否已入庫
    - 不查詢完整性與比對筆數，應該入庫前確認完整性（一次性完整寫入）

---

#### `gis_db.py` — 資料讀寫函式

- [X] 實作 `load_all_polygon_coords(conn, area_level: str = None) -> list[dict]`（取代原 `get_all_towns`）：
  - [X] `area_level` 可傳 `"county"`、`"town"` 或 `None`（不過濾，取全部）
  - [X] 以 `geopandas.read_postgis()` 從 `GIS_001` 讀出`geometry` 欄位，全部升級為 MULTIPOLYGO
  - [X] 回傳資料表 GeoDataFrame，若表不存在或查詢失敗，回傳空 GeoDataFrame 並 `logger.warning`
- [X] 實作 `upsert_gis_boundary(conn, gdf: gpd.GeoDataFrame, shp_version: str) -> bool`（取代原 `upsert_gis_polygon`）：
  - [X] `area_level` 為 `"county"` 或 `"town"`，決定 `area_id` ：
    - `"county"`：`area_id = COUNTYID`
    - `"town"`：`area_id = TOWNID`
  - [X] 先呼叫 `create_gis_table(conn)` 確保資料表存在，改為main初始化時呼叫
  - [X] 呼叫 `ensure_columns_exist()` 處理新版 SHP 可能新增的欄位
  - [X] 逐筆組裝 record dict，補入 `area_id`、`area_level`、`shp_version`
  - [X] 以 `execute_sql(auto_commit=False)` 批次寫入，任一筆失敗時 `conn.rollback()` 並回傳 `False`
  - [X] 衝突時跳過（確保版本純潔性）
  - [X] 迴圈中以 `tqdm` 顯示進度
  - [X] 全部完成後 `logger.success(f"行政區界入庫完成！版本 {shp_version}，共 {total} 筆資料。")`
  - [X] 實作特殊行政區過濾邏輯（ T28 瑪家鄉已有獨立檔案的資料補入與去重邏輯）
  - [X] 實作 `check_shp_needs_update(conn, shp_version: str) -> bool`：
    - 精簡化，僅以 `shp_version` 欄位判斷是否入庫

---

#### `gis_reader.py` — 補強

- [X] `shp_reader()` 加入 CRS 驗證：若 `gdf.crs != EPSG:4326` 則執行 `gdf.to_crs("EPSG:4326")`
  - 保留預案，不寫此語句
- [X] `shp_reader()` 加入必要欄位存在性檢查（`TOWNID` 或 `COUNTYID`、`geometry`），缺欄位時 `logger.warning`
  - 缺少 `geometry` 時返回空 GeoDataFrame
  - 缺少 `TOWNID` 或 `COUNTYID` 時返回原始 GeoDataFrame

---

#### `main_gis.py` — 選單與流程串接

- [X] 實作 `run_shp_pipeline(conn, shp_dir: str) -> None` 完整流程：
  - [X] 呼叫 `gis_db.create_gis_table(conn)` 初始化兩張資料表，改為程式初始化時呼叫
  - [X] 掃描 `shp_dir` 下所有 `.shp` 檔案，識別三個核心檔案並判斷 `area_level`：
    - `COUNTY_MOI_{民國年月日}.shp` → `area_level = "county"`
    - `TOWN_MOI_{民國年月日}.shp` → `area_level = "town"`
    - `Town_Majia_Sanhe_{民國年月日}.shp` → `area_level = "town"`
  - [X] 每個 SHP 各自呼叫 `_check_shp_needs_update(conn, shp_version, area_level)`，已完整入庫則跳過並 `logger.info`
  - [X] 呼叫 `gis_reader.shp_reader(shp_path)` 取得 GeoDataFrame
  - [X] 呼叫 `gis_db.upsert_gis_boundary(conn, gdf, shp_version, area_level)` 入庫
- [X] 重構 `main()` 選單：
  - [X] 移除「完整初始化」選項，以「更新 SHP 界線」取代（首次執行與更新邏輯相同，由 `_check_shp_needs_update` 自動判斷是否需要入庫）
  - [X] 選項規劃：
    - `1. 更新 SHP 界線`：呼叫 `run_shp_pipeline(conn, shp_dir)`
    - `2. 更新 WMS 量化`（Phase 3 實作，目前印出「尚未實作」）
    - `q. 退出`
  - [X] 從 `cli_utils` import `yes_no_menu`，替換直接 `input()` 判斷邏輯

---

#### 驗證

- [X] 單元測試 `create_gis_table(conn)`：確認 `GIS_001` 與 `GIS_metadata` 兩張表建立成功，且欄位結構符合 schema
- [X] 單元測試 `shp_reader()`：
  - 讀取 `COUNTY_MOI_1140318.shp`，確認 CRS 為 `EPSG:4326`，`COUNTYID` 欄位自然排序正確
  - 讀取 `TOWN_MOI_1140318.shp`，確認回傳 368 筆，`TOWNID` 欄位自然排序正確（`A01 < A02 < ... < W01`）
- [X] 單元測試 `upsert_gis_boundary()`：
  - 分別以 `area_level="county"` 與 `area_level="town"` 寫入
  - 以 `get_count(conn, "GIS_001")` 確認縣市筆數與鄉鎮筆數各自正確
  - 確認 `area_id` 正確填入，不組裝（county：`"A"`；town：`"A01"`）
- [X] 執行 `main_gis.py` 選單選項 1（更新 SHP 界線），確認 COUNTY 與 TOWN 三個 SHP 均可完整入庫
- [X] 確認重複執行（upsert）不產生重複列，筆數保持不變 (已在外面阻擋，不執行入庫)
- [X] `git commit`（message：`[Feature]SHP reading and geometry(Goemetry, 4325) upsert`）

---

## Phase 3：WMS 圖層下載與像素量化

### Issue 5.1：WMS 下載、地理遮罩
**狀態：** ✅ 已完成
**前置條件：** Phase 2 完成

#### `gis_db.py` — 資料讀寫函式
- [X] 實作 `log_gis_metadata(conn, area_id: str, area_level: str, classification: str, shp_version: str, save_info: dict, stage: str = "raw") -> None`：
  - [X] 以 `execute_sql()` 寫入 `GIS_metadata`，conflict 欄位為 `(area_id, classification, shp_version, stage)`
  - [X] 衝突時更新 `file_path`、`recorded_at`
- [X] 實作 `load_map_links`，讀取 `metadata_index` 並從地理圖資分類（002）中獲得對應的圖層連結
- [X] 實作 `load_all_polygon_coords`，讀取 `GIS_001` 獲得行政區界資料

---

#### `main_gis.py` — 選單與流程串接
- [X] 建立圖檔中英對照表模組 `map_name_mapping.py`
  - 四大類型：`crop_suitability_rating_map_{作物名}`，`farmland_importance_level`，`farmland_productivity_levels`，`soil_survey_{類型名}`
  - [X] 建立圖檔命名規範函式（亦為主函式） `get_or_create_map_name_en()`
- [X] 實作 `_geographic_mapping`，建立基本架構：
  - [X] 呼叫 `load_map_links` 與 `load_all_polygon_coords`，獲得基本圖層連結與行政區界資料
  - [X] 對GeoDataframe中所有 'town' 層級的 'COUNTYENG' 欄位進行補值
  - [X] 可選 SHP 版本，按照民國年月日版本號顯示
  - [X] 雙層迴圈圖層連結與行政區界資料
  - [X] 不排除 raw／mask 為全白的圖層
    - 因為存檔開銷極小，因此不再判斷無 GIS 資料的圖層
  - [X] 記錄中斷資訊，包含版本、圖層與行政區
  - [X] 將遮罩參數改為全黑 = 0，方便以後的面積計算

---

#### `gis_downloader.py` — 影像下載、比對與儲存
- [X] 修正 `replace_url_parameters` 修改參數錯誤的bug
- [X] 修正 `fetch_wms_image`，改為返回binary image，並加入重試邏輯
- [X] 修正 `_compare_and_get_status`的比對方式，改為binary image比對
- [X] 修正 `save_image`，可接受binary image或nd.array並transpose後存檔

---

#### `gis_quantifier.py` — 影像量化與色彩定義
- [X] 新增函式 `is_nearly_white` 用來檢查影像是否接近全白，改為不使用

---

### Issue 5.2：PNG 圖層矩陣量化輸出
**狀態：** 🔄 進行中
**前置條件：** Issue 5.1 完成

#### 設計決策（已確認）
- 輸出粒度：一次量化 = 一個 parquet，含所有圖層 × 所有鄉鎮 + 縣市（wide format）
- 主鍵：`area_id`（COUNTYID / TOWNID，天然唯一）
- 固定欄位：`area_id`、`area_level`、`COUNTYID`、`COUNTYNAME`、`TOWNID`（county 列填 null）、`TOWNNAME`（county 列填 null）
- 量化特徵欄位命名細節於實作 `quantify_mask()` 時另立 TODO
- 輸出檔名：`GIS_quantified_{shp_date}_{run_date}.parquet` + `GIS_quantified_{shp_date}_{run_date}.meta.json`
- `shp_date`：從 shp_version 字串提取 7 位民國年月日（沿用現有 regex）
- 量化輸入來源：查詢 GIS_metadata（stage='masked'）作為主索引，`os.path.exists()` 為前置磁碟守衛

---

#### `gis_db.py` — 查詢待量化清單

- [ ] 實作 `get_gis_metadata(conn, shp_version: str, stage: str = 'masked', area_level: str = None) -> pd.DataFrame`：
  - [ ] 查詢 GIS_metadata，條件：`shp_version`、`stage`，可選 `area_level`
  - [ ] 以 `execute_sql(..., fetch_all=True)` 取出，轉為 DataFrame 回傳
  - [ ] 若表不存在或查無資料，回傳空 DataFrame 並 `logger.warning`

---

#### `gis_quantifier.py` — 核心量化函式

- [ ] 實作 `compute_pixel_area_m2(bounds: tuple, width: int, height: int) -> float`：
  - [ ] 將 bounds（EPSG:4326）對應的矩形投影至 EPSG:3826，計算總面積（m²）
  - [ ] 除以 `width × height` 得每像素面積（m²）
  - [ ] 使用 `pyproj.Transformer` 或 `shapely` + geopandas 投影
- [ ] 實作 `quantify_mask(masked_image: np.ndarray, color_map: dict, pixel_area_m2: float) -> dict`：
  - [ ] 接收 masked image（channel-last BGR, uint8）、該圖層的 color_to_value dict、每像素面積
  - [ ] 排除背景像素（純黑 `(0,0,0)`）
  - [ ] 統計各色彩的像素數，結合 `pixel_area_m2` 計算面積
  - [ ] 依 color_to_value 的 value 型別分三路處理（scalar / categorical / list）
  - [ ] 回傳 `{feature_name: value, ...}` dict；欄位命名細節於此函式實作時另立 TODO
  - [ ] 若 masked image 內無非背景像素（全黑），回傳空 dict 並 `logger.warning`

---

#### `file_utils.py` — Parquet 讀寫

- [ ] 新增 `save_parquet(df: pd.DataFrame, file_path: str) -> str`：
  - [ ] 建立必要目錄，以 `pandas.DataFrame.to_parquet()` 存檔
  - [ ] 回傳實際儲存路徑
- [ ] 新增 `load_parquet(file_path: str) -> pd.DataFrame`：
  - [ ] 確認檔案存在，以 `pandas.read_parquet()` 讀取
  - [ ] 失敗時回傳空 DataFrame 並 `logger.error`

---

#### `main_gis.py` — 量化流程串接

- [ ] 實作 `_run_quantification(conn, shp_date: str, output_dir: str = "output") -> str`：
  - [ ] 呼叫 `get_gis_metadata(conn, shp_version=...)` 取得待量化清單，篩選 `stage='masked'`
  - [ ] `os.path.exists()` 逐筆驗證 file_path，缺失則 `logger.warning` 跳過
  - [ ] 載入 `geographic_color_metadata.json`
  - [ ] 以 `area_id` 為列，逐圖層逐行政區執行：
    - `cv2.imread()` 載入 masked image
    - `compute_pixel_area_m2()` 計算像素面積
    - `quantify_mask()` 取得特徵 dict
    - 累積至結果 DataFrame
  - [ ] 補入固定欄位（`area_id`、`area_level`、`COUNTYID`、`COUNTYNAME`、`TOWNID`、`TOWNNAME`）
  - [ ] 呼叫 `save_parquet()` 輸出主檔
  - [ ] 組裝 sidecar dict（`shp_date`、`quantified_at`、`source_record_count`、`classifications`、`latest_wms_recorded_at`、`color_metadata_mtime`），呼叫 `save_json_data()` 輸出 `.meta.json`
  - [ ] `logger.success()` 記錄輸出路徑，回傳 parquet 路徑
- [ ] 在 `main()` 選單新增選項 `3. 執行量化輸出`：
  - [ ] 互動式輸入或自動偵測 `shp_date`
  - [ ] 呼叫 `_run_quantification(conn, shp_date)`

---

#### 驗證

- [ ] 單元測試 `compute_pixel_area_m2()`：以已知 BBOX 驗算像素面積合理性
- [ ] 單元測試 `quantify_mask()`：以小型測試影像（手動建立已知色彩分布）驗證三種 value 型別路徑各自輸出正確
- [ ] 整合測試 `_run_quantification()`：
  - [ ] 確認輸出 parquet 的 row count = county 筆數 + town 筆數
  - [ ] 確認 county 列的 `TOWNID`、`TOWNNAME` 為 null
  - [ ] 確認 sidecar JSON 存在且欄位完整
- [ ] `git commit`（message：`[Feature] GIS quantification matrix output`）
- [ ] 建立 `requirements.txt`

---

## Phase 4：農業統計數據整合與前處理

### Issue 6：數據讀取、彙總策略與前處理
**狀態：** ⬜ 待開始
**前置條件：** Phase 3 完成

> Checklist 於 Phase 4 開始時補充
- [ ] 建立 `requirements.txt`

- [ ] 建立輸出檔案命名規範
  ```
  GIS_processed_{layer}_{YYYYMMDD}.csv
  GIS_processed_{layer}_{YYYYMMDD}.parquet
---

## Phase 5：特徵工程與訓練集輸出

### Issue 7：特徵工程、EDA 與訓練集輸出
**狀態：** ⬜ 待開始
**前置條件：** Phase 4 完成

> Checklist 於 Phase 5 開始時補充
- [ ] 建立 `requirements.txt`

### Issue 8：main_pipeline.py 與 argparse
**狀態：** ⬜ 待開始
**前置條件：** Issue 7 完成

> Checklist 於 Phase 5 開始時補充
- [ ] 建立 `requirements.txt`