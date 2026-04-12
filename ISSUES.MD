# moa-gis-pipeline Issues

## 狀態說明
- ⬜ 未開始
- 🔄 進行中
- ✅ 已完成

---

## Phase 1：專案基礎建設

### Issue 1：專案初始化與骨架建立
**狀態：** ✅ 進行中

#### 開發環境
- [X] 在新專案根目錄執行 `git init`
- [X] 建立 `.gitignore`
  - 排除：`.env`、`output/*`、`GIS/*.zip`、`logs/*`、`__pycache__/`、`*.pyc`、`*.parquet`、`temp.*`、
          `test.*`、`metadata/`、`old_file/`、`farmland_spatial_map/*.json`
  - 保留：`output/.gitkeep`、`logs/.gitkeep`
- [X] 建立 `.env`（含 `USERNAME`、`PASSWORD`、`DBNAME`）
- [X] 建立 `.env.example`（含欄位名稱但值為空，提交進 git）
- [ ] 建立 `requirements.txt`
  - 套件：`geopandas`、`rasterio`、`shapely`、`psycopg2`、`pandas`、`numpy`、`Pillow`、
          `vincenty`、`tqdm`、`python-dotenv`、`locale`、`json`
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
**狀態：** ✅ 待開始
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

#### `gis_quantifier.py` 填入
- [X] 從舊 `build_base_para.py` 遷移：`load_image_with_chinese_path()`
- [X] 從舊 `build_base_para.py` 遷移：`decode_png_color_value()`
  - 移除 `print()` 改用 `logger`
- [X] 新增：`save_mask_image(png, classification, country_name)`

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
**狀態：** ⬜ 待開始
**前置條件：** Issue 1 完成

#### Git 使用規範建立
- [X] 確認 `.gitignore` 涵蓋所有不應追蹤的檔案
- [X] 在 GitHub 建立 public repository
- [X] 推送 Issue 1、Issue 2 的成果作為初始 commit
- [X] 建立 commit message 規範：
  - `[Setup]` 環境與骨架
  - `[Refactor]` 重構與遷移
  - `[Feature]` 新功能
  - `[Fix]` 錯誤修正
  - `[Data]` 資料相關變動

#### 資料庫備份策略
- [ ] 確認 `pg_dump` 可正常執行
  ```
  pg_dump -U {USERNAME} -d {DBNAME} -f backup_{date}.sql
  ```
- [ ] 建立備份目錄結構（加入 `.gitignore`）
  ```
  backups/
  └── moa_opendata/    ← GIS 資料與農業統計數據共用同一資料庫，統一備份
  ```
- [ ] 決定備份頻率（建議：每個 Phase 完成後手動執行一次）
- [ ] 確認備份檔案會同步到外部儲存（外接硬碟或雲端）

#### `geographic_color_metadata.json` 定位確認
- [ ] 確認此檔案進入 git 追蹤（設定檔性質，幾 KB，適合 git 管理）
- [ ] 確認此檔案**不入庫**
  - 原因：`color_to_value` 為異質結構，入庫複雜度高、收益低；git 已提供版本追蹤能力
- [ ] 在 `gis_db.py` 的 docstring 中記錄此決策

#### 輸出檔案備份策略
- [ ] 確認 `output/` 目錄下的 CSV/Parquet 不進入 git
- [ ] 建立輸出檔案命名規範
  ```
  GIS_processed_{layer}_{YYYYMMDD}.csv
  GIS_processed_{layer}_{YYYYMMDD}.parquet
  ```
- [ ] 確認輸出檔案會定期備份到外部儲存

#### 驗證
- [ ] GitHub repository 可正常推送與拉取
- [ ] `pg_dump` 執行成功，備份檔案可正常還原（`psql` 測試）
- [ ] `git log` 確認有清晰的 commit 歷史

---

## Phase 2：地理界線解析與入庫

### Issue 4：SHP 讀取與地理資料入庫
**狀態：** ⬜ 待開始
**前置條件：** Phase 1 全部完成

> Checklist 於 Phase 2 開始時補充

---

## Phase 3：WMS 圖層下載與像素量化

### Issue 5：WMS 下載、地理遮罩與量化輸出
**狀態：** ⬜ 待開始
**前置條件：** Phase 2 完成

> Checklist 於 Phase 3 開始時補充

---

## Phase 4：農業統計數據整合與前處理

### Issue 6：數據讀取、彙總策略與前處理
**狀態：** ⬜ 待開始
**前置條件：** Phase 3 完成

> Checklist 於 Phase 4 開始時補充

---

## Phase 5：特徵工程與訓練集輸出

### Issue 7：特徵工程、EDA 與訓練集輸出
**狀態：** ⬜ 待開始
**前置條件：** Phase 4 完成

> Checklist 於 Phase 5 開始時補充

### Issue 8：main_pipeline.py 與 argparse
**狀態：** ⬜ 待開始
**前置條件：** Issue 7 完成

> Checklist 於 Phase 5 開始時補充