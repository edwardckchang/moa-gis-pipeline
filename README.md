---

# moa-gis-pipeline

`moa-gis-pipeline` 是一個用於處理地理資訊系統 (GIS) 資料與農業開放數據的自動化處理流程。本專案旨在整合 SHP 圖層解析、WMS 影像下載量化，以及農業統計數據的前處理，最終產出用於機器學習的特徵訓練集。

> [!CAUTION]
> **開發狀態：** 本專案目前處於 **Phase 1 (專案基礎建設)** 階段。核心模組僅完成骨架與部分功能遷移，尚未達到可正式運作（Production-ready）的程度。

---

## 🏗️ 專案架構

本專案採用模組化設計，將資料獲取、處理與儲存分離，以提高維護性：

| 模組名稱 | 職責說明 |
| :--- | :--- |
| **`main_gis.py`** | Phase 1-3 的主程式入口，管理 GIS 資料流。 |
| **`database_manager.py`** | 底層資料庫連線與 SQL 執行邏輯（使用 SQLAlchemy/psycopg2）。 |
| **`gis_reader.py`** | 負責 SHP 檔案讀取、自然排序與地理座標轉換。 |
| **`gis_downloader.py`** | 處理 WMS 圖層下載與重試機制。 |
| **`gis_quantifier.py`** | 負責將地理影像像素值量化為具備業務意義的數值。 |
| **`gis_db.py`** | 處理 GIS 專屬的資料庫邏輯（如 Schema 推導）。 |
| **`analysis_pipeline.py`** | 資料清洗、轉換與特徵工程處理。 |
| **`utils.py` / `file_utils.py`** | 通用字串處理、JSON 檔案讀寫等工具函式。 |

---

## 📈 當前進度

專案規劃分為五個階段，目前的開發進度如下：

### **Phase 1：專案基礎建設** 🔄 (進行中)
- [x] 初始化開發環境與目錄結構。
- [x] 重構並建立基礎工具模組 (`utils`, `database_manager`, `file_utils` 等)。
- [x] 建立所有核心功能模組的空白骨架。
- [ ] 驗證資料庫連線與 `requirements.txt` 環境相容性。

### **Phase 2 - Phase 5** ⬜ (待開始)
- **Phase 2：** 地理界線解析與入庫 (SHP Processing)。
- **Phase 3：** WMS 圖層下載與像素量化。
- **Phase 4：** 農業統計數據整合與前處理。
- **Phase 5：** 特徵工程與訓練集輸出 (Final Pipeline)。

---

## 🛠️ 開發環境

- **Language:** Python 3.9
- **Database:** PostgreSQL (PostGIS 支援)
- **Main Libraries:** `geopandas`, `rasterio`, `shapely`, `pandas`, `psycopg2`

---

## 📝 備註
詳細的開發工作細項與 Issue 追蹤請參閱 `ISSUES.md`。