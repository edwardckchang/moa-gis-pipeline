"""
main_pipeline.py
================
整合入口：串接 GIS 流水線（main_gis.py）與統計分析流水線（analysis_pipeline.py），
支援 argparse 排程 / 自動化執行。

【對話五實作，對話五之前不動此檔案】

職責範圍：
  - 接收 CLI 參數，決定執行哪些階段（全跑 / 僅統計前處理 / 僅特徵工程）
  - 依序呼叫各模組的流水線函式，不在此處寫業務邏輯
  - 統一管理資料庫連線生命週期（moa_gis 與 moa_opendata 兩條連線）
  - 執行結束後輸出訓練集路徑，供排程系統或下游腳本取用

與 main_gis.py 的分工：
  main_gis.py   — 層級一（SHP 入庫）、層級二（WMS 量化）的互動式 CLI 選單入口，
                  有人值守時手動操作使用
  main_pipeline.py — 層級三（統計前處理 + 特徵工程）至訓練集輸出的自動化入口，
                     設計為無人值守、可排程執行

argparse 參數規劃（對話五確認）：
  --mode        執行模式，可選：
                  full        完整執行（統計前處理 + 特徵工程 + 輸出）
                  preprocess  僅執行統計前處理與 JOIN（對話四範疇）
                  features    僅執行特徵工程與訓練集輸出（需前置 preprocess 已完成）
  --gis-output  GIS 量化結果檔案路徑（.csv 或 .parquet），
                預設掃描 output/ 目錄取最新一筆
  --output-dir  訓練集輸出目錄，預設 output/
  --format      輸出格式，可選 csv / parquet，預設 parquet
  --log-level   logger 輸出層級，預設 15（NOTICE）

典型呼叫方式：
  # 完整執行（排程）
  python main_pipeline.py --mode full --format parquet

  # 僅重跑特徵工程（GIS 量化與統計前處理已完成）
  python main_pipeline.py --mode features --gis-output output/GIS_processed_002_farmland_20250101.parquet

TODO（Phase 5 — 對話五）：
  - [ ] 實作 parse_args()：建立 argparse.ArgumentParser，定義上述所有參數
  - [ ] 實作 resolve_gis_output(output_dir, layer)：
        若未指定 --gis-output，自動掃描 output/ 目錄，
        以 glob + sorted 取最新一筆對應 layer 的量化結果檔案
  - [ ] 實作 run_preprocess_pipeline(conn_moa, gis_output_path)：
        呼叫 analysis_pipeline 的前處理函式序列：
          load_agricultural_statistics → summarize_large_dataset
          → join_gis_and_statistics → handle_missing_values
          → encode_categorical → normalize_features
  - [ ] 實作 run_feature_pipeline(df)：
        呼叫 analysis_pipeline 的特徵工程函式序列：
          engineer_features → run_eda → export_training_set
  - [ ] 實作 main()：
        1. parse_args()
        2. setup_logging()
        3. 依 --mode 決定執行路徑
        4. 建立並管理兩條 DB 連線（moa_gis、moa_opendata）
        5. 依序呼叫 run_preprocess_pipeline() / run_feature_pipeline()
        6. logger.success() 記錄輸出檔案路徑與執行時間
  - [ ] if __name__ == "__main__": 呼叫 main()

相依模組（預期）：
  argparse, os, glob
  database_manager（connect_conn, disconnect_conn）
  analysis_pipeline（所有前處理與特徵工程函式）
  logs_handle（setup_logging, logger）
"""

# TODO（Phase 5）：實作 parse_args() -> argparse.Namespace
# TODO（Phase 5）：實作 resolve_gis_output(output_dir, layer) -> str
# TODO（Phase 5）：實作 run_preprocess_pipeline(conn_moa, gis_output_path) -> pd.DataFrame
# TODO（Phase 5）：實作 run_feature_pipeline(df) -> str
# TODO（Phase 5）：實作 main() -> None
