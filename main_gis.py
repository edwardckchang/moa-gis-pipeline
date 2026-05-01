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
import os, sys
def _patch_proj_data() -> None:
    """
    覆寫 PROJ_DATA / PROJ_LIB，避免 PostGIS 安裝後污染 rasterio 的 GDAL。
    PostGIS 安裝程式會將舊版 proj.db 路徑寫入系統環境變數，
    必須在任何 rasterio / GDAL import 發生之前覆寫，否則 GDAL 已鎖定錯誤路徑。
    """
    if sys.platform != "win32":
        return
    import pathlib
    candidates = [
        pathlib.Path(sys.prefix) / "Lib" / "site-packages" / "rasterio" / "proj_data",
        pathlib.Path(sys.prefix) / "Lib" / "site-packages" / "pyogrio" / "proj_data",
    ]
    import site
    for sp in [site.getusersitepackages()] + site.getsitepackages():
        candidates.append(pathlib.Path(sp) / "rasterio" / "proj_data")
        candidates.append(pathlib.Path(sp) / "pyogrio" / "proj_data")

    for proj_dir in candidates:
        if (proj_dir / "proj.db").exists():
            os.environ["PROJ_DATA"] = str(proj_dir)
            os.environ["PROJ_LIB"]  = str(proj_dir)
            print(f"[PROJ patch] 使用：{proj_dir}")
            return

    print("[PROJ patch] ⚠️ 找不到有效的 proj.db，PROJ 衝突未解決")
_patch_proj_data()
from database_manager import connect_conn, disconnect_conn
from logs_handle import logger, setup_logging
from gis_downloader import replace_url_parameters, fetch_wms_image, save_image
from dotenv import load_dotenv, dotenv_values
import pandas as pd
from gis_reader import get_width_height_from_geographic_mapping, png_geographic_mapping, shp_reader
import re, time
from gis_db import create_gis_table, load_all_polygon_coords, check_shp_needs_update, upsert_gis_boundary
from gis_db import load_map_links, log_gis_metadata, check_gis_exists
from cli_utils import yes_no_menu
from utils import init_checkpoint, checkpoint as cp
from map_name_mapping import get_or_create_map_name_en
from file_utils import load_json_data, save_json_data

def _save_checkpoint(shp_ver: str, map_name_en: str, area_id: str) -> None:
    data = load_json_data(CHECKPOINT_PATH)
    data[shp_ver] = {
        "map_name_en": map_name_en,
        "area_id":     area_id,
        "status":      "interrupted"
    }
    save_json_data(data)

def _load_checkpoint(shp_ver: str) -> tuple[dict, bool]:
    data = load_json_data(CHECKPOINT_PATH)
    entry = data.get(shp_ver, {})
    if not entry:
        return {}, True
    if entry.get("status") == "completed":
        choice = yes_no_menu(f"版本 {shp_ver} 上次已完整完成，是否從頭開始")
        return {}, choice
    logger.notice(
        f"偵測到版本 {shp_ver} 中斷點："
        f"圖層={entry['map_name_en']}, area_id={entry['area_id']}，"
        f"將從此位置繼續。"
    )
    return entry, True

def _clear_checkpoint(shp_ver: str) -> None:
    """僅更新指定版本的 status，其他版本不受影響。"""
    data = load_json_data(CHECKPOINT_PATH)
    if shp_ver in data:
        data[shp_ver]["status"] = "completed"
    else:
        data[shp_ver] = {"status": "completed"}
    save_json_data(data)

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
    """
    def _log_result_message(result, area_id, map_name_en, shp_version, save_info, stage="raw"):
        if result == True:
            logger.logs(f"{area_id} : {stage} / {map_name_en} / {shp_version} 紀錄完成 ({save_info['status']})")
        elif result == False:
            logger.logs(f"{area_id} : {stage} / {map_name_en} / {shp_version} 紀錄失敗")
    # 1. 取得 WMS 圖層 URL
    map_links = load_map_links(conn)
    if not map_links or len(map_links) == 0:
        logger.warning("無法取得 WMS 圖層連結，中止 _geographic_mapping。")
        return
    # 2. 從 GIS_001 取得鄉鎮 geometry，回傳GeoDataframe
    all_polygon_gdf = load_all_polygon_coords(conn)
    if all_polygon_gdf is None or all_polygon_gdf.empty:
        logger.error("GIS_001 無資料，請先完成層級一 SHP 入庫。")
        return
    # 2.1 對GeoDataframe中所有 'town' 層級的 'COUNTYENG' 欄位進行補值
    if all_polygon_gdf is not None and not all_polygon_gdf.empty:
        # 1. 建立縣市英文對照表 (確保只抓有值的)
        county_map = all_polygon_gdf[
            (all_polygon_gdf['area_level'] == 'county') & 
            (all_polygon_gdf['COUNTYENG'].notna()) & 
            (all_polygon_gdf['COUNTYENG'] != '')
        ].set_index('COUNTYID')['COUNTYENG'].to_dict()
        # 2. 填補 TOWN 等級缺失的 COUNTYENG
        # 邏輯：如果 area_level 是 town 且 COUNTYENG 為空，就根據 COUNTYID 去對照表抓值
        town_mask = (all_polygon_gdf['area_level'] == 'town') & (all_polygon_gdf['COUNTYENG'].isna())
        all_polygon_gdf.loc[town_mask, 'COUNTYENG'] = all_polygon_gdf.loc[town_mask, 'COUNTYID'].map(county_map)        
        logger.info("已完成記憶體內 COUNTYENG 補值 (僅限本次執行)")

    # 2.2 選擇 SHP 版本
    # 從 shp_version 欄位萃取民國年月日（7位數），去重後排序供選單使用
    SHP_ver_pattern = re.compile(r'.*?(\d{7})(?!\d)')
    SHP_vers = sorted(set(
        m.group(1)
        for v in all_polygon_gdf["shp_version"].dropna().unique()
        for m in [SHP_ver_pattern.search(str(v))]
        if m
    ), reverse=True)  # 最新版本排最前

    if not SHP_vers:
        logger.error("GIS_001 中無法解析任何有效的 SHP 版本號，請確認入庫資料。")
        return
    # 互動式選擇 SHP 版本
    while True:
        print("\n請選擇要下載的 SHP 版本（輸入 q 退出）：")
        if len(SHP_vers) == 1:
            shp_ver = SHP_vers[0]
            break
        for n, ver in enumerate(SHP_vers, start=1):
            print(f"  {n}. {ver}")
        choice = input("\n請輸入選項：").strip().lower()

        if choice == 'q':
            return

        if not choice.isdigit():
            print("請輸入數字選項。")
            continue

        choice_int = int(choice)
        if 1 <= choice_int <= len(SHP_vers):
            shp_ver = SHP_vers[choice_int - 1]
            break
        else:
            print(f"請輸入 1 到 {len(SHP_vers)} 之間的數字。")

    logger.notice(f"已選擇 SHP 版本：{shp_ver}")
    # 過濾掉選定以外的版本
    mask_ver = all_polygon_gdf["shp_version"].str.contains(shp_ver, regex=False, na=False)
    all_polygon_records = all_polygon_gdf.loc[mask_ver].to_dict("records")

    checkpoint, choice = _load_checkpoint(shp_ver)
    if not choice:
        return
    resume_map  = checkpoint.get("map_name_en", "")   # 空字串 = 從頭開始
    resume_area = checkpoint.get("area_id", "")
    skip_map  = bool(resume_map)
    skip_area = False

    if not all_polygon_records:
        logger.error(f"版本 {shp_ver} 篩選後無任何行政區資料，請確認入庫狀況。")
        return

    # 3. 雙層迴圈：圖層 × 鄉鎮
    for map_name, map_link in map_links.items():
        map_name_en = get_or_create_map_name_en(map_name)
        if not map_name_en:
            logger.error(f"圖層 '{map_name}' 無對應英文名稱，跳過。")
            continue
        if skip_map: # 讀取中斷點，跳過上次完成的圖層
            if map_name_en != resume_map:
                input(f"Checkpoint 跳過圖層：{map_name_en}")
                continue
            # 找到 checkpoint 圖層，停止跳過
            skip_map  = False
            skip_area = bool(resume_area)

        for polygon_record in all_polygon_records:
            area_id = polygon_record.get("area_id")
            if skip_area: # 讀取中斷點，跳過上次完成的行政區
                if area_id != resume_area:
                    logger.debug(f"Checkpoint 跳過行政區：{area_id}")
                    continue
                # 找到 resume_area，這筆已完成，跳過本身從下一筆開始
                input(f"Checkpoint 最後跳過的行政區：{area_id}")
                skip_area = False
                continue
            shp_version = polygon_record.get("shp_version")
            area_level = polygon_record.get("area_level")
            geometry_obj = polygon_record.get("geometry")
            if area_level == "town":
                countyeng = polygon_record.get("COUNTYENG") or ""
                towneng   = polygon_record.get("TOWNENG") or ""
                if not countyeng or not towneng:
                    logger.error(
                        f"COUNTYENG 或 TOWNENG 為空 "
                        f"（COUNTYENG={countyeng!r}, TOWNENG={towneng!r}），"
                        f"跳過 area_id={area_id}。"
                    )
                    continue
                region_en = f"{countyeng}_{towneng}"
            elif area_level == "county":
                countyeng = polygon_record.get("COUNTYENG") or ""
                if not countyeng:
                    logger.error(f"COUNTYENG 為空，跳過 area_id={area_id}。")
                    continue
                region_en = countyeng
            else:
                logger.error(f"未知 area_level='{area_level}'，跳過 area_id={area_id}。")
                continue

            if geometry_obj is None:
                logger.error(f"geometry 為 None，跳過 area_id={area_id}。")
                continue
            region_en = region_en.replace(" ", "_")
            cp(f"{[shp_version, area_level, area_id, geometry_obj, region_en]}")
            
            # 3b. 計算 BBOX 與請求尺寸
            bounds = geometry_obj.bounds  # (minx, miny, maxx, maxy) == (lon_min, lat_min, lon_max, lat_max)
            width, height = get_width_height_from_geographic_mapping(bounds, target_res)
            new_params = {
                "BBOX":   f"{bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}",
                "WIDTH":  width,
                "HEIGHT": height,
            }

            # 3c. 組裝完整 URL
            request_url = replace_url_parameters(map_link, new_params)

            # 3d. 下載 raw 影像
            raw_image = fetch_wms_image(request_url)
            if raw_image is None:
                logger.warning(f"WMS 下載失敗，跳過 area_id={area_id} / {map_name_en}。")
                continue
            # 3e. 落地 raw 影像
            ori_save_info = save_image(raw_image, map_name_en, region_en, shp_version, "raw")
            # 3f. raw 落地失敗 → 跳過此行政區
            if ori_save_info['status'] == "error":
                logger.logs(f"{area_id} 地區的 raw 圖層 {map_name_en} 版本 {shp_version} 落地失敗，跳過。")
                continue
            cp(ori_save_info)
            if ori_save_info['status'] != "unchanged":                
                logger.debug(f"{area_id} 地區的 raw 圖層 {map_name_en} 版本 {shp_version} 落地完成，狀態為 {ori_save_info['status']}")
            # 3g. raw 未變動 → 確認 masked 是否已存在，存在則跳過遮罩
            if ori_save_info['status'] == "unchanged":                
                if check_gis_exists(conn, ori_save_info, area_id, map_name_en, shp_version, "masked"):
                    continue
                else:
                    ori_save_info['status'] = "created"
            result = log_gis_metadata(conn, area_id, area_level, map_name_en, shp_version, ori_save_info)
            _log_result_message(result, area_id, map_name_en, shp_version, ori_save_info)

            # 3h. 執行地理遮罩
            masked_image, _ = png_geographic_mapping(geometry_obj, bounds, raw_image, target_res)
            if masked_image is None:
                logger.logs(f"{area_id} 地區的 raw 圖層 {map_name_en} 版本 {shp_version} 遮罩失敗，跳過。")
                continue

            # 3i. 落地 masked 影像（channel transpose 由 save_image 內部統一處理）
            mask_save_info = save_image(masked_image, map_name_en, region_en, shp_version, "masked")
            cp(mask_save_info)
            if mask_save_info['status'] == "error":
                logger.logs(f"{area_id} 地區的 masked 圖層 {map_name_en} 版本 {shp_version} 落地失敗，跳過。")
                continue
            if mask_save_info['status'] != "unchanged":                
                logger.debug(f"{area_id} 地區的 masked 圖層 {map_name_en} 版本 {shp_version} 落地完成，狀態為 {mask_save_info['status']}")
            if mask_save_info['status'] == "unchanged":
                if check_gis_exists(conn, mask_save_info, area_id, map_name_en, shp_version, "masked"):
                    continue
                mask_save_info['status'] = "created"
            result = log_gis_metadata(conn, area_id, area_level, map_name_en, shp_version,
                mask_save_info, stage="masked")
            _log_result_message(result, area_id, map_name_en, shp_version, mask_save_info, stage="masked")
            _save_checkpoint(shp_ver, map_name_en, area_id)
        logger.success(f"圖層 '{map_name_en}' 全部行政區處理完畢。")
        for n in range(5, 0, -1):
            print(f"--- {n} 秒後繼續 ---", end="\r")
            time.sleep(1)
        print(" " * 20, end="\r") 
    _clear_checkpoint(shp_ver)

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
    setup_logging(level=20)
    load_dotenv()
    config = dotenv_values()
    DB_USER     = config.get("DB_USER")
    DB_PASSWORD = config.get("DB_PASSWORD")
    DB_NAME     = config.get("DB_NAME")
    TABLE_GIS_001 = "GIS_001" # 如果要修改表格名稱，要在 gis_db 全域新增表格名稱參數
    TABLE_GIS_METADATA = "GIS_metadata" # 如果要修改表格名稱，要在 gis_db 全域新增表格名稱參數
    SHP_DIR = "boundaries"
    """
    raw_image_path = os.path.join("output", "wms_images", shp_version, "raw", classification)
    masked_image_path = os.path.join("output", "wms_images", shp_version, "masked", classification)
    如果要修改分布圖存檔資料結構，可以將 gis_downloader.save_image() 參數拉到 _geographic_mapping 或此處做設定。
    """    
    CHECKPOINT_PATH = "output/geographic_mapping_checkpoint.json"
    conn = connect_conn(DB_USER, DB_PASSWORD, DB_NAME)
    init_checkpoint(True, False)
    # 以下開始為 _geographic_mapping 的語句逐步測試
    maplinks = load_map_links(conn)

    if not conn:
        logger.error("資料庫連線失敗，程式終止。")
        raise SystemExit(1)
    create_gis_table(conn)
    target_res = 100 #設定解析度： 像素(px) / target_res(m)
    try:
        main(conn, target_res)
    finally:
        disconnect_conn(conn)
