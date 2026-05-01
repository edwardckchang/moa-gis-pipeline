"""
gis_downloader.py
=================
負責 WMS 圖層的 HTTP 請求與影像下載，所有影像在記憶體中處理，不落地至磁碟。

職責範圍：
  - WMS URL 的查詢參數替換（BBOX、WIDTH、HEIGHT 等）
  - 發送 HTTP GET 請求取得 PNG 回應
  - 將 response bytes 解碼為 numpy.ndarray（cv2 BGR 格式），供下游直接使用
  - 將 raw / masked 影像依版本策略落地至輸出目錄，並回傳儲存結果供 metadata 記錄

設計原則：
  - 此模組只負責「下載與儲存」，不處理任何像素量化或地理遮罩邏輯
  - 下載結果直接以 numpy.ndarray 回傳，不寫入任何暫存檔
  - WMS URL 來源由呼叫端（main_gis.py）從 moa_opendata 資料庫取得後傳入，
    此模組不持有任何 URL 或連線狀態

典型呼叫流程（層級二，main_gis.geographic_mapping()）：
  1. 從 GIS_001 取得鄉鎮 geometry，計算 bounds
  2. 從 moa_opendata 取出 WMS 基底 URL
  3. 呼叫 replace_url_parameters() 注入 BBOX、WIDTH、HEIGHT
  4. 呼叫 fetch_wms_image() 取得 numpy.ndarray（raw）
  5. 呼叫 save_image(..., stage="raw") 落地，取得含路徑與狀態的 save_info dict
  6. 若 raw 未變動且 masked 檔案已存在，跳過遮罩步驟
  7. 傳入 gis_reader.png_geographic_mapping() 執行地理遮罩
  8. 呼叫 save_image(..., stage="masked") 落地，取得 save_info dict
  9. 呼叫端依兩個 save_info 寫入 GIS_metadata

TODO（Phase 3 — 層級二）：
  - [ ] 在 fetch_wms_image() 加入重試機制（建議 3 次，間隔指數退避），
        處理 WMS 服務偶發的 timeout 或 5xx 錯誤
  - [ ] 在 fetch_wms_image() 加入 Content-Type 驗證，
        確認 response 為 image/png 而非 XML 錯誤訊息（WMS 服務錯誤時仍回傳 200）

相依模組：
  requests, cv2（OpenCV）, numpy, datetime, os
  logs_handle（統一 logger）
"""

import requests
import cv2
import numpy as np
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from logs_handle import logger
from datetime import datetime
import os, time

def replace_url_parameters(original_url: str, new_params: dict) -> str:
    """
    替換（或新增）URL 查詢字串中的指定參數，其餘參數保持不變。
    使用 urllib.parse 解析後重組，安全處理多值參數（parse_qs 回傳 list）。
    新值一律轉為字串後以單元素 list 覆蓋，確保不產生重複參數。
    特別處理大小寫不敏感的覆蓋，避免同時出現 bbox 與 BBOX 的情況。

    Args:
        original_url (str): 原始 WMS URL，例如：
            "https://talis.moa.gov.tw/wms?SERVICE=WMS&VERSION=1.1.1
             &REQUEST=GetMap&BBOX=120,23,121,24&WIDTH=1000&HEIGHT=1100"
        new_params (dict): 欲替換或新增的查詢參數，key / value 均為字串或可轉為字串的型別，
            例如 {"BBOX": "120.1,23.1,120.9,23.9", "WIDTH": 857, "HEIGHT": 922}。

    Returns:
        str: 替換參數後的完整 URL 字串，scheme / host / path / fragment 均保持原樣。

    Notes:
        - parse_qs() 對同名參數會回傳 list，此函式以單元素 list 覆蓋，
          若原始 URL 有重複參數（罕見），替換後只會保留一個值
        - WMS 規範的參數名稱大小寫依服務而定，替換時以傳入的 key 為準，
          不做大小寫正規化

    Examples:
        >>> replace_url_parameters(
        ...     "https://example.com/wms?BBOX=0,0,1,1&WIDTH=100",
        ...     {"BBOX": "120,23,121,24", "WIDTH": 857}
        ... )
        'https://example.com/wms?BBOX=120%2C23%2C121%2C24&WIDTH=857'
    """
    parsed_url = urlparse(original_url)
    # query_params 格式為 {key: [value1, value2]}
    query_params = parse_qs(parsed_url.query)

    for new_key, new_value in new_params.items():
        # 尋找現有參數中，是否存在大小寫不同但名稱相同的 key
        # 例如：new_key 是 'BBOX'，而 query_params 裡有 'bbox'
        target_keys = [k for k in query_params.keys() if k.lower() == new_key.lower()]
        
        # 刪除所有衝突的舊 Key (不論大小寫)
        for k in target_keys:
            del query_params[k]
        
        # 插入新的參數值
        query_params[new_key] = [str(new_value)]

    # 重新組裝 URL
    updated_query = urlencode(query_params, doseq=True)
    return urlunparse(parsed_url._replace(query=updated_query))

def fetch_wms_image(url: str) -> "bytes | None":
    """
    向 WMS 服務發送 GET 請求，直接回傳 PNG bytes（不落地）。
    具備指數退避重試機制。
    Args:
        url (str): 完整的 WMS GetMap 請求 URL，通常由 replace_url_parameters() 組裝。

    Returns:
        bytes | None:
          - 成功：PNG bytes
          - 失敗（HTTP 錯誤、timeout、解碼失敗）：None

    Notes:
        - timeout 固定為 30 秒；WMS 服務對大範圍 BBOX 的回應可能較慢，
          Phase 3 可根據實測調整或改為動態計算
        - WMS 服務在參數錯誤時可能回傳 HTTP 200 但內容為 XML 錯誤訊息，
          此情況下 None，已由 logger.error 記錄
    """
    max_retries = 3
    for n in range(max_retries + 1):
        try:
            response = requests.get(url, timeout=10)            
            # 如果是 5xx 錯誤，觸發 HTTPError 進入重試邏輯
            # 4xx 錯誤（如 BBOX 錯誤）通常重試也沒用，所以直接 raise
            if 500 <= response.status_code < 600:
                response.raise_for_status()            
            # 其他狀態碼（如 200, 4xx）
            response.raise_for_status()
            # 檢查 Content-Type 確保是影像
            content_type = response.headers.get('Content-Type', '')
            if 'image' not in content_type:
                logger.error(f"下載內容不是影像。Type: {content_type}, URL: {url}")
                return None
            return response.content
        except (requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
            # 判斷是否還有重試機會
            if n < max_retries:
                wait_time = 2 ** n * 5  # 指數退避：1s, 2s, 4s
                logger.warning(
                    f"WMS 請求失敗 ({e})，準備進行第 {n+1} 次重試，"
                    f"等待 {wait_time} 秒... (URL: {url})"
                )
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"已達到最大重試次數 ({max_retries})，放棄下載：{url}")
                return None                
        except Exception as e:
            # 對於非暫時性的錯誤（如 URL 格式錯誤、連線被拒絕等），直接放棄不重試
            logger.error(f"下載 WMS 影像時發生未預期錯誤（不重試）：{e}")
            return None            
    return None

def _compare_and_get_status(file_path: str, new_bytes: bytes) -> str:
    """
    透過 Binary (Bytes) 比對檔案內容。
    
    Args:
        file_path: 磁碟上的既有檔案路徑
        new_bytes: 準備寫入的新影像 Binary 內容
    Returns:
        "created" | "unchanged" | "updated"
    """
    # 1. 檔案不存在，視為新增
    if not os.path.exists(file_path):
        return "created"
    try:
        # 2. 讀取既有檔案的 Binary 內容
        with open(file_path, 'rb') as f:
            existing_bytes = f.read()
        # 3. 快速比對：先比對長度 (檔案大小)，若不同則必定不同
        if len(existing_bytes) != len(new_bytes):
            return "updated"
        # 4. 深度比對：比對內容 (也可以用 hashlib 做雜湊比對更穩)
        # 這裡直接用 == 運算子比對 bytes 是效能最高且最直接的做法
        if existing_bytes == new_bytes:
            return "unchanged"        
        return "updated"
    except Exception as e:
        logger.warning(f"_compare_and_get_status：讀取既有檔案失敗 ({e})，視為需更新。")
        return "updated"

def save_image(
    image_data: "np.ndarray | bytes",
    classification: str,
    region_name_en: str,
    shp_version: str,
    stage: str
) -> dict:
    """
    將 raw 或 masked 影像依版本策略落地至輸出目錄，並回傳儲存結果 dict
    供 main_gis.py 寫入 GIS_metadata。

    目錄結構與版本策略：
        output/wms_images/{classification}/raw/
            {region_name_en}.png          ← 永遠只保留最新版，內容有變時覆蓋
        output/wms_images/{classification}/masked/{shp_version}/
            {region_name_en}.png          ← 以 SHP 檔名為版本目錄，自然隔離不同界線版本

        raw  策略：覆蓋更新。落地前以 np.array_equal() 比對既有檔案：
                     - 初次落地：status="created"
                     - 內容不同：status="updated"（覆蓋）
                     - 內容相同：status="unchanged"（跳過寫入）
        masked 策略：版本隔離。同一 shp_version 目錄下邏輯與 raw 相同；
                     raw 狀態為 "unchanged" 且 masked 檔案已存在時，
                     呼叫端可直接跳過遮罩步驟（不需呼叫本函式）。

    Args:
        image (np.ndarray): 待儲存的影像。
            - raw：fetch_wms_image() 回傳的 channel-last (H, W, 3) ndarray
            - masked：png_geographic_mapping() 回傳的 channel-first (C, H, W) ndarray，
              本函式內部自動偵測並 transpose 為 channel-last 再編碼
        classification (str): 圖層英文名稱，同時作為一級子目錄，
            例如 "farmland_importance_level"、"soil_ph"。
            來源：圖檔中英對照表，由 main_gis.py 查表後傳入。
        region_name_en (str): 行政區英文名稱，作為檔名（不含副檔名），
            例如 "da_an_dist"、"zhongli_dist"。
            來源：行政區中英對應表，由 main_gis.py 查表後傳入。
        shp_version (str): SHP 檔名（不含副檔名），作為 masked 的版本目錄名稱，
            例如 "TOWN_MOI_1140318"。
            stage="raw" 時此參數不影響路徑，但仍需傳入（可傳空字串）。
        stage (str): 儲存階段，必須為 "raw" 或 "masked"。

    Returns:
        dict: 儲存結果，供呼叫端判斷後續流程並寫入 GIS_metadata，結構如下：
            {
              "status": str,   # "created" | "updated" | "unchanged" | "error"
              "path":   str,   # 實際落地的完整檔案路徑；error 時為空字串
            }
        呼叫端使用範例（main_gis.geographic_mapping()）：
            ori_save_info = save_image(raw_img, ..., stage="raw")
            if ori_save_info["status"] == "error": ...
            if ori_save_info["status"] == "unchanged" and masked_exists: skip_mask()
            mask_save_info = save_image(masked_img, ..., stage="masked")
            # 依兩個 save_info 寫入 GIS_metadata

    Notes:
        - channel-first 偵測規則：image.ndim == 3 且 image.shape[0] <= 4，
          判定為 (C, H, W)，執行 transpose((1, 2, 0)) 後再編碼
        - 中英對照表（classification、region_name_en）的查表邏輯在 main_gis.py，
          本函式只接受已轉換的英文字串
    """
    # 1. 組合落地路徑
    dir_path = os.path.join("output", "wms_images", shp_version, stage, classification)
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, f"{region_name_en}.png")
    # 2. 根據輸入類型處理
    if isinstance(image_data, bytes):
        final_bytes = image_data
    elif isinstance(image_data, np.ndarray):     
        success, buf = cv2.imencode('.png', image_data)
        if not success:
            logger.error(f"cv2.imencode 失敗：{file_path}")
            return {"status": "error", "path": ""}
        final_bytes = buf.tobytes()
    else:
        logger.error(f"檔案{dir_path}的格式錯誤，無法儲存。")
        return {"status": "error", "path": ""}
    # --- 執行 Binary 比對 ---
    status = _compare_and_get_status(file_path, final_bytes)
    # 3. 落地儲存
    if status == "unchanged":
        logger.debug(f"[unchanged] {stage} 影像二進位內容一致，跳過：'{file_path}'")
        return {"status": "unchanged", "path": file_path}
    try:
        with open(file_path, 'wb') as f:
            f.write(final_bytes)
        logger.notice(f"[{status}] {stage} 影像已儲存：'{file_path}'")
        return {"status": status, "path": file_path}
    except Exception as e:
        logger.error(f"儲存影像失敗：{e}")
        return {"status": "error", "path": ""}
