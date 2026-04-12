import requests
import cv2
import numpy as np
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from logs_handle import logger

def replace_url_parameters(original_url, new_params):
    """
    替換網址中的查詢參數。

    Args:
        original_url (str): 原始網址字串。
        new_params (dict): 包含要替換的新參數的字典。

    Returns:
        str: 替換參數後的網址字串。
    """
    parsed_url = urlparse(original_url)
    query_params = parse_qs(parsed_url.query)
    for key, value in new_params.items():
        query_params[key] = [str(value)]
    updated_query = urlencode(query_params, doseq=True)
    return urlunparse(parsed_url._replace(query=updated_query))

def fetch_wms_image(url):
    """
    執行下載並將 bytes 轉為 cv2 影像矩陣（不落地）。
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # 使用你提到的直覺做法：bytearray -> imdecode
        file_bytes = np.frombuffer(response.content, dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error("影像解碼失敗，回傳內容可能不是有效的 PNG")
            return None
        return image
    except Exception as e:
        logger.error(f"下載 WMS 影像時發生錯誤: {e}")
        return None
