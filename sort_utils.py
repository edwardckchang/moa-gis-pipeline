from tqdm import tqdm
from utils import get_comparison_columns, parse_date_string
from logs_handle import logger
import locale
import json

def sort_list_of_dictionaries(data: list[dict], sort_keys: list[list[str, str]]) -> list[dict]:
    """
    使用排序鍵排序字典列表中的字典。

    Args:
        data (list[dict]): 要排序的字典列表。
        sort_keys (list[list[str, str]]): 排序鍵的列表，每個元素是一個元組 (鍵名, 排序類型)。
                                            排序類型可以是 'date_asc', 'text_asc', 'number_asc'。

    Returns:
        list[dict]: 排序後的字典列表。
    """
    # 設定中文語系以支援中文排序
    try:
        locale.setlocale(locale.LC_ALL, 'zh_TW.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'zh_TW')
        except locale.Error:
            logger.warning("無法設定中文語系，中文排序可能不準確。")

    def get_sort_value(item, key, sort_type):
        value = item.get(key)
        if sort_type == 'date_asc':
            try:
                str_value = str(value).strip()
                return parse_date_string(str_value)
            except (TypeError, ValueError) as e:
                logger.error(f"日期值 '{value}' 類型錯誤或解析失敗 ({e})，不進行排序。")
                return False
        elif sort_type == 'text_asc':
            # 定義優先級映射表，用於「上中下」或「前中後」等概念的排序
            priority_map = {
                "上": 1, "前": 1, "一": 1,
                "中": 2, "二": 2,
                "下": 3, "後": 3, "三": 3
            }
            str_value = str(value).strip() if value is not None else ""
            
            # 檢查是否在優先級映射表中
            if str_value in priority_map:
                return priority_map[str_value]
            
            # 使用 locale.strxfrm 進行中文排序
            return locale.strxfrm(str(value)) if value is not None else ""
        elif sort_type == 'number_asc':
            try:
                return float(value)
            except (ValueError, TypeError):
                return value # 如果無法轉換為數字，則返回原始值
        return value

    # 步驟 1: 計算每個項目的排序鍵，並顯示進度條
    # 這裡我們創建一個新的列表，包含 (排序鍵, 原始項目) 的元組
    # 這是可以疊代並顯示進度的地方
    items_with_keys = []
    logger.info("正在進行資料排序...")
    if isinstance(sort_keys, str):
        try:
            # 如果是字串，嘗試解析為 list
            sort_keys = json.loads(sort_keys)
        except Exception as e:
            logger.error(f"轉換排序鍵時發生錯誤: {e}")
            return []
    with tqdm(total=len(data)) as pbar:
        for item in data:
            # custom_sort_key 函數的邏輯保持不變，只是現在我們明確地迭代資料
            try:
                key = tuple(get_sort_value(item, k, st) for k, st in sort_keys)
            except Exception as e:
                logger.error(f"排序時發生錯誤: {e}")
                return []
            items_with_keys.append((key, item))
            pbar.update(1) # 更新進度條

    # 步驟 2: 根據預先計算好的排序鍵進行排序
    # 這個 sorted() 操作本身無法直接顯示進度，但它會比每次都計算鍵快
    sorted_items_with_keys = sorted(items_with_keys, key=lambda x: x[0])
    logger.notice("資料排序完成。")

    # 步驟 3: 從排序後的元組中提取原始項目
    return [item for key, item in sorted_items_with_keys]

def _interactive_sort_data(data: list[dict], predefined_sort_keys: list[list[str, str]] = None) -> tuple[list[dict], list[list[str, str]]]:
    """
    引導使用者選擇排序欄位和類型，然後對資料進行排序。
    如果提供了 predefined_sort_keys，則直接使用這些鍵進行排序。

    Args:
        data (list[dict]): 要排序的字典列表。
        predefined_sort_keys (list[list[str, str]], optional): 預定義的排序鍵列表。
                                                                 如果提供，則跳過互動式選擇。

    Returns:
        tuple[list[dict], list[list[str, str]]]: 排序後的字典列表和實際使用的排序鍵。
    """
    if not data:
        logger.warning("資料為空，無法進行排序。")
        return [], []

    sort_keys = []
    if predefined_sort_keys:
        sort_keys = predefined_sort_keys
        logger.notice(f"已套用預設排序鍵。")
    else:
        selected_columns = get_comparison_columns(data)
        
        if not selected_columns:
            logger.warning("未選擇任何排序欄位，回傳原始資料。")
            return data, []

        for column in selected_columns:
            while True:
                sort_type_choice = input(f"請為 '{column}' 選擇排序類型 (1: 日期從舊到新, 2: 文字從a到z, 3: 數字從小到大): ")
                if sort_type_choice == '1':
                    sort_type = 'date_asc'
                    break
                elif sort_type_choice == '2':
                    sort_type = 'text_asc'
                    break
                elif sort_type_choice == '3':
                    sort_type = 'number_asc'
                    break
            sort_keys.append([column, sort_type])
            logger.info(f"已為 '{column}' 設定排序類型為 '{sort_type}'")
    if not sort_keys:
        logger.warning("未選擇任何排序欄位，回傳原始資料。")
        return data, []
    logger.notice(f"最終排序鍵：{sort_keys}")
    return sort_list_of_dictionaries(data, sort_keys), sort_keys
