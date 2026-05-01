import json
import os
from logs_handle import logger
from typing import Union

def load_json_data(file_path: str):
    """
    從指定 JSON 檔案讀取資料。
    Args:
        file_path (str): JSON 檔案的路徑。
    Returns:
        list[dict] | None: 解析後的字典列表或是字典，如果發生錯誤則返回 None。
    """
    try:
        file_path = os.path.normpath(file_path)
    except:
        pass
    if not os.path.exists(file_path):
        logger.error(f"檔案 '{file_path}' 不存在。")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"解析 JSON 檔案 '{file_path}' 時發生錯誤: {e}")
        return None
    except Exception as e:
        logger.error(f"讀取檔案 '{file_path}' 時發生錯誤: {e}")
        return None

    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        if not isinstance(data, dict):
            logger.error(f"檔案 '{file_path}' 中的資料格式不符合預期 (字典列表或單一字典)。")
            return None    
    return data

def save_json_data(data: Union[list, dict], file_path: str) -> str:
    """
    將 JSON 資料儲存到指定路徑。
    Args:
        data (list): 要儲存的 JSON 資料。
        file_path (str): 儲存的路徑。
    Returns:
        str: 儲存後的檔案路徑。
    """
    # 如果 file_path 包含目錄，則創建目錄
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return file_path