"""
main_gis.py
===========
GIS 專屬 CLI 選單入口，管理層級一（SHP 入庫）與層級二（WMS 下載量化）的執行流程。

職責範圍：
  層級一（Phase 2）— SHP 行政界線解析與入庫：
    - 掃描 boundaries/ 目錄下的所有 SHP 檔案
    - 與 GIS_metadata 比對，判斷是否有新版本需要入庫（新檔名即新版本）
    - 呼叫 gis_reader.shp_reader() 讀取 GeoDataFrame
    - 呼叫 gis_db.save_gis_bounds_to_postgresql() 寫入 GIS_001

  層級二（Phase 3）— WMS 圖層下載與遮罩：
    - 提供 geographic_mapping() 作為層級二的核心入口函式
    - 管理圖檔中英對照表與行政區中英對應表的載入
    - 串接 gis_downloader、gis_reader、gis_db 的各函式
    - 依 save_image() 回傳的 status 決定是否跳過遮罩，並寫入 GIS_metadata

與 main_pipeline.py 的分工：
  main_gis.py       — 有人值守的互動式 CLI，層級一到層級二
  main_pipeline.py  — 無人值守的自動化排程入口，層級三至訓練集輸出

連線管理：
  - moa_gis（本專案）：conn，在 __main__ 建立，貫穿整個執行期，以 try/finally 確保關閉
  - moa_opendata（WMS URL 來源）：在 geographic_mapping() 內建立第二條連線，查完即關閉
  - conn 以參數傳入各 gis_db 函式；moa_opendata 連線不對外暴露

中英對照表管理：
  本模組持有兩個對照表，供 geographic_mapping() 查表後傳入 save_image()：
    圖檔中英對照表：WMS 圖層中文名稱 → 英文分類名稱（作為目錄名與 classification 欄位）
    行政區中英對應表：TOWNID → 英文行政區名稱（作為檔名的 region_name_en）
        可自行政區界表中取出
  行政區英文名稱同時寫入 GIS_001.TOWNNAME_EN，對照表來源為 TOWNID 對應英文名稱的 JSON，
  格式與 geographic_color_metadata.json 相同，以 file_utils.load_json_data() 載入。
  圖檔中英對照表來源待 Phase 3 確認（hardcode dict 或 JSON）。

TODO（Phase 2 — 層級一）：
  - [X] 實作 scan_shp_files(shp_dir) -> list[str]：
        只有一行，不需要建函式
  - [ ] 實作 check_shp_needs_update(conn, shp_version) -> bool：
        查詢 GIS_metadata，判斷此 shp_version 是否已完整入庫；
        若未入庫或入庫不完整則回傳 True
  - [ ] 實作 run_shp_pipeline(conn, shp_path) -> bool：
        呼叫 shp_reader() → save_gis_bounds_to_postgresql()，
        完成後寫入 GIS_metadata，回傳是否成功
  - [ ] 重構 main() 選單（見 main() TODO）

TODO（Phase 3 — 層級二）：
  - [ ] 建立行政區中英對應表 JSON（key: TOWNID，value: 英文名稱）
        自行政區界表中取出
  - [ ] 確認圖檔中英對照表來源與格式
        （含作物名稱翻譯：crop_suitability_rating_map_{作物名}、soil_survey_{類型名}）
  - [ ] 實作 load_map_links(conn_moa) -> dict：
        建立 moa_opendata 第二條連線，取出 WMS 圖層 URL，
        回傳 {圖層中文名稱: URL} dict，查完關閉連線
  - [ ] 實作 load_all_polygon_coords(conn) -> dict：
        從 GIS_001 取出所有鄉鎮的 TOWNID 與 geometry_wkt，
        以 shapely.wkt.loads() 還原為 Shapely Polygon，
        回傳 {TOWNID: polygon} dict
  - [ ] 補完 geographic_mapping() 的完整實作（見函式內 TODO）

相依模組：
  database_manager（connect_conn, disconnect_conn, execute_sql）
  gis_reader（shp_reader, get_width_height_from_geographic_mapping, png_geographic_mapping）
  gis_downloader（replace_url_parameters, fetch_wms_image, save_image）
  gis_db（create_gis_table, save_gis_bounds_to_postgresql,
          get_all_towns, log_gis_metadata
  file_utils（load_json_data）
  cli_utils（yes_no_menu）
  logs_handle（setup_logging, logger）
"""

from database_manager import connect_conn, disconnect_conn, execute_sql
from logs_handle import logger, setup_logging
from gis_downloader import replace_url_parameters, fetch_wms_image, save_image
from dotenv import load_dotenv, dotenv_values
import pandas as pd
from gis_reader import get_width_height_from_geographic_mapping, png_geographic_mapping, shp_reader
import os
from gis_db import create_gis_table, get_gis_metadata, load_all_polygon_coords, check_shp_needs_update, upsert_gis_boundary
from gis_db import load_map_links
from cli_utils import yes_no_menu
from utils import init_checkpoint, Checkpoint

# TODO（Phase 3）：確認來源格式後載入
# 格式：{"農地重要性等級": "farmland_importance_level", ...}
# 含作物名稱與土壤類型的翻譯對照


def _run_shp_pipeline(conn, shp_dir: str) -> None:
    """
    層級二鄉鎮市區界經緯度資料入庫
    執行流程：
      1. 呼叫 gis_db.create_gis_polygon_table() 與 gis_db.create_gis_metadata_table() 初始化 GIS_001 與 GIS_metadata 資料表
        已移到程式初始化時執行
      2. 掃描 boundaries/ 目錄下的所有 SHP 檔案，核心檔案有三個：COUNTY_MOI_{民國年月日}，Town_Majia_Sanhe，TOWN_MOI_{民國年月日}
        其中Town_Majia_Sanhe需要解壓縮後自行加上後綴'_{民國年月日}'
      3. 呼叫 gis_db.check_shp_needs_update，判斷是否有新版本需要入庫（新檔名即新版本）
      4. 呼叫 gis_reader.shp_reader() 讀取 GeoDataFrame
      5. 呼叫 gis_db.upsert_gis_boundary() 寫入 GIS_001
    Args:
        conn: moa_gis 資料庫的 psycopg2 connection 物件。
    """
    shp_file_list = [f for f in os.listdir(shp_dir) if f.endswith(".shp") and ("county" in f.lower() or "town" in f.lower())]
    for shp_file in shp_file_list:
        shp_path = os.path.join(shp_dir, shp_file)
        shp_version = os.path.splitext(shp_file)[0]
        if not check_shp_needs_update(conn, shp_version):
            continue
        gdf = shp_reader(shp_path)
        upsert_gis_boundary(conn, gdf, shp_version)

def _geographic_mapping(conn, target_res: int = 100) -> None:
    """
    層級二核心入口：依序執行 WMS 圖層下載、地理遮罩、落地存檔，並記錄執行結果至 GIS_metadata。

    前置條件：
      - GIS_001 已完成入庫（層級一完成）
      - 模組層級的圖檔中英對照表

    執行流程：
      1. 取出 WMS 圖層 URL dict
      2. 從 GIS_001 取出所有縣市鄉鎮 geometry，還原為 {region_name_en: Shapely Polygon} dict
        area_level = town
            region_name_en = COUNTYENG + "_" + TOWNENG
        area_level = county
            region_name_en = COUNTYENG
      3. 雙層迴圈（外層：圖層，內層：鄉鎮）：
           a. 既然官方表格有，在名稱在取出時就拿到了
           b. polygon.bounds 取 BBOX
           c. replace_url_parameters() 注入 BBOX / WIDTH / HEIGHT
           d. fetch_wms_image() 下載 raw 影像
           e. save_image(..., stage="raw") 落地，取得 ori_save_info dict
           f. ori_save_info["status"] == "error"：logger.error，continue 跳過此鄉鎮
           g. ori_save_info["status"] == "unchanged"：
                查詢 GIS_metadata 確認 masked 檔案路徑存在且檔案在磁碟上存在，
                兩者均成立則 continue 跳過遮罩；否則繼續執行（補救遮罩）
           h. png_geographic_mapping() 執行地理遮罩，取得 masked ndarray
           i. save_image(..., stage="masked") 落地，取得 mask_save_info dict
           j. log_gis_metadata() 分別寫入 raw 與 masked 兩筆執行紀錄

    Args:
        conn: moa_gis 資料庫的 psycopg2 connection 物件。
        shp_version (str): 本次使用的 SHP 版本（不含副檔名），例如 "TOWN_MOI_1140318"，
            作為 masked 目錄名稱與 GIS_metadata 的 shp_version 欄位。
        target_res (int): WMS 請求的地面解析度（公尺/像素），預設 100。

    Returns:
        None。影像落地至 output/wms_images/，執行紀錄寫入 GIS_metadata。

    Notes:
        - 飛地：部分鄉鎮的 geometry 為 MultiPolygon（例如離島行政區），
          png_geographic_mapping() 接受 MultiPolygon，rasterio.mask() 可正常處理，
          但 bounds 涵蓋整個外包矩形，下載面積較實際陸地大，需於 Phase 3 確認行為
        - 圖層命名規範（圖檔中英對照表的 value）：
            農地重要性等級      → farmland_importance_level
            農地生產力等級      → farmland_productivity_levels
            XX適栽性等級分布圖  → crop_suitability_rating_map_{作物名}
            土壤XX / 母岩性質   → soil_survey_{類型名}
          作物名稱與土壤類型需翻譯對照，待 Phase 3 建立

    TODO（Phase 3）：
        - [ ] 實作流程步驟 1：呼叫 load_map_links(conn)
        - [ ] 實作流程步驟 2：呼叫 load_all_polygon_coords(conn)
        - [ ] 補完步驟 3f/3g 的 status 判斷邏輯
              （"unchanged" 時查詢 GIS_metadata 並以 os.path.exists() 確認磁碟檔案存在）
        - [ ] 補完步驟 3h/3i → png_geographic_mapping() → save_image(..., stage="masked")
        - [ ] 補完步驟 3j → log_gis_metadata() 寫入兩筆紀錄（raw + masked）
        - [ ] 建立圖檔中英對照表（含作物名稱與土壤類型的翻譯）
        - [ ] 移除舊專案 GLOBAL_METADATA 邏輯，改為 load_map_links()
    """
    # 1. 取得 WMS 圖層 URL
    map_links = load_map_links(conn)

    # 2. 從 GIS_001 取得鄉鎮 geometry，回傳GeoDataframe
    all_polygon = load_all_polygon_coords(conn)
    # 組裝字典
    
    # 3. 雙層迴圈：圖層 × 鄉鎮
    for map_name, map_link in map_links.items():
        map_name_en = 圖檔中英對照表.get(map_name)
        if not map_name_en:
            logger.warning(f"圖層 '{map_name}' 無對應英文名稱，跳過。")
            continue

        for polygon_coords in all_polygon:

            region_en = polygon_coords.get("COUNTYENG") + "_" + polygon_coords.get("TOWNENG")

            # 3b. 計算 BBOX
            bounds = polygon_coords.bounds
            width, height = get_width_height_from_geographic_mapping(bounds, target_res)
            new_params = {
                "BBOX":   f"{bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}",
                "WIDTH":  width,
                "HEIGHT": height,
            }

            # 3d. 下載 raw 影像
            ori_region_pic = fetch_wms_image(
                replace_url_parameters(map_link, new_params)
            )

            # 3e. 落地 raw
            ori_save_info = save_image(
                ori_region_pic, map_name_en, region_en, shp_version, "raw"
            )

            # 3f. error：跳過此鄉鎮
            if ori_save_info["status"] == "error":
                logger.error(f"raw 儲存失敗，跳過 {townid} / {map_name}。")
                continue

            # 3g. unchanged：確認 masked 是否已存在，存在則跳過遮罩
            # TODO（Phase 3）：查詢 GIS_metadata + os.path.exists() 雙重確認
            # if ori_save_info["status"] == "unchanged":
            #     masked_path = gis_db.get_masked_path(conn, townid, map_name_en, shp_version)
            #     if masked_path and os.path.exists(masked_path):
            #         continue

            # 3h. 執行地理遮罩
            mask_region_pic, _ = png_geographic_mapping(
                polygon_coords, bounds, ori_region_pic, target_res
            )

            # 3i. 落地 masked
            # TODO（Phase 3）：取消下方註解並補完
            # mask_save_info = save_image(
            #     mask_region_pic, map_name_en, region_en, shp_version, "masked"
            # )

            # 3j. 寫入 GIS_metadata
            # TODO（Phase 3）：呼叫 gis_db.log_gis_metadata(conn, record)
            # 兩筆：raw（ori_save_info）與 masked（mask_save_info）


def main(conn, target_res: int = 100) -> None:
    """
    GIS 工具的 CLI 主選單，提供 SHP 入庫與 WMS 量化的互動式操作入口。

    選單規劃：
      1. 僅更新 SHP 界線
           僅執行層級一（SHP 入庫），跳過 WMS 下載
      2. 僅更新 WMS 量化
           跳過 SHP 入庫，直接執行層級二（WMS 下載 + 遮罩），
           shp_version 從 GIS_metadata 查詢最新已入庫版本
      q. 退出

    Args:
        conn: moa_gis 資料庫的 psycopg2 connection 物件，
            由 __main__ 建立並傳入，__main__ 的 finally 確保關閉。

    Notes:
        - pd.set_option 顯示設定保留，供後續 CLI 顯示 DataFrame 使用
        - 目前函式內的舊專案殘留（yes_no_menu、make_AUTO_YES、update_by_metadata 等）
          在 Phase 2 實作時全部替換

    TODO（Phase 2）：
        - [ ] 從 cli_utils 匯入 yes_no_menu，替換現有 input() 互動邏輯
        - [ ] 實作選單選項 1：
              scan_shp_files → check_shp_needs_update() → run_shp_pipeline()
              → geographic_mapping(conn, shp_version)
        - [ ] 實作選單選項 2：
              scan_shp_files() → check_shp_needs_update() → run_shp_pipeline()
        - [ ] 實作選單選項 3：
              從 GIS_metadata 查詢最新 shp_version → geographic_mapping(conn, shp_version)
    """
    print("歡迎使用 moa-gis-pipeline GIS 工具。")
    pd.set_option('display.unicode.east_asian_width', True)
    pd.set_option('display.width', 120)
    pd.set_option('display.max_colwidth', 60)
    pd.set_option('display.colheader_justify', 'left')

    # TODO（Phase 2）：以下為規劃骨架，實作時替換舊專案殘留邏輯
    while True:
        print("\n請選擇操作：")
        print("1. 更新 SHP 界線")
        print("2. 更新 WMS 量化")
        print("q. 退出")
        choice = input("請輸入您的選擇：").strip().lower()
        if choice == 'q':
            print("感謝使用，程式即將退出。")
            break
        elif choice == '1':
            _run_shp_pipeline(conn, SHP_DIR)
        elif choice == '2':
            _geographic_mapping(conn, target_res)
            print("此功能尚未實作。")
        else:
            print("無效的選擇，請重新輸入。")


if __name__ == "__main__":
    setup_logging(level=15)
    load_dotenv()
    config = dotenv_values()
    shp_dir = os.path.normpath("boundaries")
    DB_USER     = config.get("DB_USER")
    DB_PASSWORD = config.get("DB_PASSWORD")
    DB_NAME     = config.get("DB_NAME")
    TABLE_GIS_001 = "GIS_001"
    TABLE_GIS_METADATA = "GIS_metadata"
    SHP_DIR = "boundaries"
    conn = connect_conn(DB_USER, DB_PASSWORD, DB_NAME)
    init_checkpoint(True, True)
    if not conn:
        logger.error("資料庫連線失敗，程式終止。")
        raise SystemExit(1)
    create_gis_table(conn)
    target_res = 100
    try:
        main(conn, target_res)
    finally:
        disconnect_conn(conn)
