# moa-gis-pipeline Roadmap

台灣農業部農業資料開放平台 — 地理圖資處理與數據分析工具

## 狀態說明
- ⬜ 未開始
- 🔄 進行中
- ✅ 已完成

---

## Milestone 1 — 專案基礎建設

- 🔄 清理並重構繼承自舊專案的模組
  - utils.py（精簡＋補充）
  - database_manager.py（架構改動：conn 由呼叫端持有）
  - 新增 sort_utils.py、file_utils.py、cli_utils.py
- 🔄 建立新專案模組骨架
  - gis_db.py、gis_reader.py、gis_downloader.py
  - gis_quantifier.py、analysis_pipeline.py
  - main_gis.py、main_pipeline.py
- 🔄 將現有可用函式填入骨架模組
- 🔄 建立 git 版本控制與資料備份策略
- ✅ 確認專案資料夾結構

---

## Milestone 2 — 地理界線解析與入庫

- ⬜ SHP 讀取與處理（gis_reader.py）
- ⬜ 地理資料存入資料庫（資料表命名待對話二確認）
- ⬜ GIS 專屬資料庫操作模組（gis_db.py 部分實作）
- ⬜ CLI 選單入口（main_gis.py 基本選單）

---

## Milestone 3 — 局部圖資下載與量化

- ⬜ WMS 圖層請求與 PNG 下載（gis_downloader.py）
- ⬜ 地理遮罩與像素量化（gis_quantifier.py）
- ⬜ 量化結果輸出為檔案（output/ CSV / Parquet）
- ⬜ GIS 專屬資料庫操作模組（gis_db.py 完整實作）
- ⬜ 色彩對應表維持 JSON 管理，進入 git 追蹤
    （不入庫，理由：異質結構不適合關聯式資料表，
     git 已能提供版本追蹤能力）

---

## Milestone 4 — 農業統計數據整合與前處理

- ⬜ 從 moa_opendata 讀取農業統計數據集
- ⬜ 確認各資料集結構與可用欄位
- ⬜ 設計大型資料集彙總策略
- ⬜ 數據前處理（缺值、標準化、編碼）
- ⬜ 與 GIS 量化結果合併

---

## Milestone 5 — 特徵工程與訓練集輸出

- ⬜ 衍生特徵設計
- ⬜ EDA
- ⬜ 訓練集輸出（.csv / .parquet）
- ⬜ 整合入口完成（main_pipeline.py，含 argparse 排程支援）

---

## 資料來源
- 行政界線 SHP：https://data.gov.tw/dataset/7441
- WMS 圖層：talis.moa.gov.tw
- 農業統計數據：moa_opendata 資料庫
- 色彩對應表：geographic_color_metadata.json（JSON 管理，不入庫）

## 環境
- Python 3.9 / PostgreSQL 17 + PostGIS 3.5.3
- 輸出格式：.csv / .parquet