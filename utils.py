import json
import os, re
from logs_handle import logger
import datetime
import pandas as pd
from typing import Optional, List
from datetime import datetime
import inspect

MAX_NAME_LENGTH_BYTES = 63 # PostgreSQL NAMEDATALEN - 1 (for null terminator)
MAX_NAME_LENGTH_CHARS = 21 # 21 中文字 = 63 位元組 (假設 UTF-8 一個中文佔 3 位元組)

def _convert_roc_to_ad(roc_year_str):
    """
    將民國年份字串轉換為西元年份整數。
    例如：'111' -> 2022
    """
    try:
        roc_year = int(roc_year_str)
        return roc_year + 1911
    except ValueError:
        logger.warning(f"無法將民國年份 '{roc_year_str}' 轉換為數字。")
        return None # 返回 None 表示轉換失敗

def parse_date_string(str_value: str):
    """
    解析多種日期時間格式，包括民國日期。
    """
    # 嘗試解析民國僅有年份 (例如: 111)
    str_value = str_value.replace(' ', '') # 移除字串中間的空格
    roc_year_only_match = re.match(r'^(\d{1,3})$', str_value)
    if roc_year_only_match:
        roc_year_str = roc_year_only_match.group(1)
        ad_year = _convert_roc_to_ad(roc_year_str)
        if ad_year is not None:
            try:
                return datetime.datetime(ad_year, 1, 1) # 視為該年份的1月1日
            except ValueError as e:
                logger.warning(f"解析民國年份 '{str_value}' (轉換為 '{ad_year}') 時發生錯誤: {e}，將其視為最小日期。")
                return datetime.datetime.min
        else:
            logger.warning(f"民國年份 '{roc_year_str}' 轉換失敗，無法解析日期 '{str_value}'，將其視為最小日期。")
            return datetime.datetime.min

    # 嘗試解析民國日期格式 (例如: 111.10.17 或 99/9/9)
    roc_full_date_match = re.match(r'^((\d{1,3})[./](\d{1,2})[./](\d{1,2}))$', str_value)
    if roc_full_date_match:
        roc_year_str, month_str, day_str = roc_full_date_match.groups()[1:]
        ad_year = _convert_roc_to_ad(roc_year_str)
        if ad_year is not None:
            try:
                # 重新組合成西元格式字串並解析
                ad_date_str = f"{ad_year}/{month_str}/{day_str}"
                return datetime.datetime.strptime(ad_date_str, "%Y/%m/%d")
            except ValueError as e:
                logger.warning(f"解析民國日期 '{str_value}' (轉換為 '{ad_date_str}') 時發生錯誤: {e}，將其視為最小日期。")
                return datetime.datetime.min
        else:
            logger.warning(f"民國年份 '{roc_year_str}' 轉換失敗，無法解析日期 '{str_value}'，將其視為最小日期。")
            return datetime.datetime.min

    # 嘗試解析多種西元日期時間格式
    for fmt in ["%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d", "%Y-%m-%d", "%Y%m%d", "%Y-%m", "%Y", "%Y年"]:
        try:
            return datetime.datetime.strptime(str_value, fmt)
        except ValueError:
            continue
    
    # 如果所有格式都無法解析，則返回 datetime.min 以確保其排在所有有效日期之前
    logger.warning(f"無法解析日期 '{str_value}'，將其視為最小日期。")
    return datetime.datetime.min

def clean_table_name(title: str) -> str:
    """
    清理標題，生成符合 PostgreSQL 命名規範的表格名稱，並限制長度。
    移除特殊字元和特定中文詞彙。
    """
    # 移除 " - 農業資料開放平臺"
    cleaned_title = title.replace(" - 農業資料開放平臺", "")
    # 移除 "農地空間圖"
    cleaned_title = cleaned_title.replace("農地空間圖", "")
    # 移除所有非字母、數字、底線的字元
    cleaned_title = re.sub(r'[^\w]', '', cleaned_title)

    # 限制長度
    if len(cleaned_title) > MAX_NAME_LENGTH_CHARS:
        # 嘗試按字元截斷
        truncated_title = cleaned_title[:MAX_NAME_LENGTH_CHARS]
        # 再次檢查位元組長度，確保不會因為中文字元導致超長
        while len(truncated_title.encode('utf-8')) > MAX_NAME_LENGTH_BYTES:
            truncated_title = truncated_title[:-1] # 每次移除一個字元直到符合位元組限制
        logger.warning(f"表格名稱 '{title}' 超過 {MAX_NAME_LENGTH_CHARS} 字元或 {MAX_NAME_LENGTH_BYTES} 位元組，已截斷為 '{truncated_title}'")
        return truncated_title
    
    return cleaned_title

def display_dataframe(df: pd.DataFrame, title: str, sort_columns: Optional[List[str]] = None, show_index: bool = False, transpose: bool = False, truncate=True) -> pd.DataFrame:
    """
    統一 DataFrame 的顯示邏輯，並可選地根據指定欄位進行排序和顯示序號。
    Args:
        df (pd.DataFrame): 要顯示的 DataFrame。
        title (str): 顯示的標題。
        sort_columns (Optional[List[str]]): 可選的欄位名稱列表，用於排序。
                                            如果提供，將按這些欄位從小到大排序。
        show_index (bool): 是否顯示 DataFrame 的序號（列號）。預設為 False。
    Returns:
        pd.DataFrame: 如果顯示了序號，則返回包含序號的 DataFrame；否則返回原始 DataFrame。
    """
    def _truncate_chinese_text(text, max_length=15):
        """截斷中文字符串，並在末尾添加 '...' 如果超過最大長度。"""
        if isinstance(text, str):
            if len(text) > max_length:
                return text[:max_length] + '...'
        return text
    if transpose:
        df = df.T
        logger.debug(f"DataFrame 已轉置:\n{df}")
    if sort_columns:
        try:
            df = df.sort_values(by=sort_columns, ascending=True).reset_index(drop=True)
        except KeyError as e:
            logger.warning(f"排序欄位錯誤: {e}。請檢查欄位名稱是否正確。")
        except Exception as e:
            logger.error(f"排序時發生未知錯誤: {e}")
    if df is None or df.empty or len(df) == 0:
        logger.warning(f"沒有 {title} 可供顯示。")
        return pd.DataFrame()        
    # 創建一個用於顯示的 DataFrame 副本，並對其進行截斷
    df_display_copy = df.copy()
    if truncate:
        for col in df_display_copy.select_dtypes(include=['object']).columns:
            df_display_copy[col] = df_display_copy[col].apply(_truncate_chinese_text)
    if show_index:
        # 如果需要顯示序號，則將索引作為新的一列添加到 DataFrame 中
        df_display_copy = df_display_copy.reset_index()
    # Determine segment size based on dataframe length
    df_length = len(df)
    if df_length <= 25:
        segment_size = df_length
    elif df_length <= 100:
        segment_size = 20
    else:
        segment_size = 50
    num_segments = (df_length + segment_size - 1) // segment_size # Calculate total number of segments
    if num_segments > 1 and not transpose:
        print(f"\n{title} (共 {df_length} 筆資料，每頁顯示 {segment_size} 筆):")
        df_to_segment = df_display_copy # Default to display copy
        if show_index:
            df_to_segment = df_display_copy.reset_index() # Use df with index if show_index is True
        first_pass =True
        for i in range(num_segments):
            if first_pass:
                pass
            else:
                input("按 Enter 繼續...")
                first_pass = False
            start_index = i * segment_size
            end_index = min((i + 1) * segment_size, df_length)
            print(f"\n--- 第 {i+1}/{num_segments} 頁 (資料 {start_index + 1} 到 {end_index} 筆) ---")
            print(f"\n{df_to_segment[start_index:end_index].to_string(index=False)}")
    else:
        print(f"\n{title} (共 {df_length} 筆資料):")
        if not transpose:
            print(f"\n{df_display_copy.to_string(index=False)}")
        else:
            print(f"\n{df_display_copy.to_string(index=True, header=False)}")
    if show_index:
        return df.reset_index() # Return original DataFrame with index if show_index is True
    else:
        return df # Return original DataFrame
    
def select_row_by_index(df: pd.DataFrame, 
                        prompt_message: str = "請選擇項目: ", 
                        sort_columns: list = None, 
                        transpose: bool = False, 
                        truncate: bool = True) -> dict:
    """
    提示使用者輸入序號（從 1 開始），並返回該行資料。
    """
    if df is None or len(df) == 0:
        logger.warning("DataFrame 為空，無法選擇行。")
        return {}

    # 1. 處理排序並重置索引
    if sort_columns:
        try:
            df = df.sort_values(by=sort_columns, ascending=True).reset_index(drop=True)
        except Exception as e:
            logger.error(f"排序失敗: {e}")
            return {}

    # 2. 建立一個從 1 開始的「序號」欄位用於顯示
    # 我們複製一份 df 以免污染原始資料，並將 index 轉為 1~N
    display_df = df.copy()
    display_df.insert(0, '序號', range(1, len(display_df) + 1))

    # 3. 顯示 DataFrame
    # 注意：這裡傳入 display_df，且因為我們已經手動加了 '序號'，
    # 建議在 display_dataframe 中關閉 Pandas 原生的 index 顯示（如果該函式支援）
    display_dataframe(display_df, prompt_message, None, False, transpose, truncate)

    while True:
        choice_str = input(f"{prompt_message}(輸入 1-{len(df)} 或 'q' 退出): ").strip()
        
        if choice_str.lower() == 'q':
            logger.info("取消選擇操作。")
            return {}
            
        try:
            # 使用者輸入的是 1-based 序號
            user_choice = int(choice_str)
            
            # 4. 將使用者輸入的序號轉回 0-based 索引 (choice_index = user_choice - 1)
            choice_index = user_choice - 1
            
            if 0 <= choice_index < len(df):
                # 使用 iloc 抓取原始 df (不含暫時序號欄位) 的資料
                selected_row = df.iloc[choice_index]
                return selected_row.to_dict()
            else:
                logger.warning(f"超出範圍，請輸入 1 到 {len(df)} 之間的數字。")
        except ValueError:
            logger.warning("無效輸入，請輸入數字序號或 'q'。")
        except Exception as e:
            logger.error(f"選擇行時發生錯誤: {e}")
            return {}

def remove_duplicates_from_list_of_dicts(data: list[dict]) -> list[dict]:
    """
    從字典列表中移除重複的字典。
    重複的判斷基於字典的內容，不考慮順序。
    Args:
        data (list[dict]): 包含可能重複字典的列表。
    Returns:
        list[dict]: 移除了重複字典的列表，保持原始順序。
    """
    unique_data = []
    seen_items = set()
    for item in data:
        # 將字典轉換為 JSON 字串作為唯一識別碼，確保鍵的順序一致
        item_identifier = json.dumps(item, sort_keys=True)
        if item_identifier not in seen_items:
            seen_items.add(item_identifier)
            unique_data.append(item)
    return unique_data

def get_comparison_columns(json_data_raw: list[dict]) -> list[str]:
    """
    引導使用者選擇用於比較的欄位。
    Args:
        json_data_raw: 原始 JSON 資料，預期為字典列表。
    Returns:
        comparison_columns: 選擇的欄位名稱列表。
    """
    comparison_columns = []
    if not json_data_raw:
        logger.warning("json_data_raw 為空，無法選擇比較欄位。")
        return comparison_columns

    available_keys = list(json_data_raw[0].keys())
    selected_indices = set()

    print("\n請選擇要用於比較的欄位 (輸入數字，輸入 'q' 結束)：")
    while True:
        for i, key in enumerate(available_keys):
            status = "(已選擇)" if i in selected_indices else ""
            print(f"{i+1}. {key}: {json_data_raw[0].get(key, 'N/A')} {status}")

        choice = input("請輸入您的選擇：")
        if choice.lower() == 'q':
            break
        
        try:
            index = int(choice) - 1
            if 0 <= index < len(available_keys):
                if index not in selected_indices:
                    selected_indices.add(index)
                    comparison_columns.append(available_keys[index])
                    logger.notice(f"已選擇 '{available_keys[index]}'")
                else:
                    logger.warning("該欄位已被選擇，請勿重複選擇。")
        except ValueError:
            logger.warning("無效的輸入，請輸入數字編號或 'q' 離開。")
        logger.notice(f"目前已選擇的比較欄位：{comparison_columns}")
    
    logger.notice(f"已選擇的比較欄位：{comparison_columns}")
    return comparison_columns

def get_filename_from_path(file_path: str) -> str:
    """
    從檔案路徑中提取檔案名稱，並限制長度。
    Args:
        file_path (str): 檔案的完整路徑。
    Returns:
        str: 截斷後的檔案名稱。
    """
    base_name = os.path.basename(file_path)
    filename_without_ext, file_extension = os.path.splitext(base_name)

    if len(filename_without_ext) > MAX_NAME_LENGTH_CHARS:
        truncated_filename = filename_without_ext[:MAX_NAME_LENGTH_CHARS]
        # 再次檢查位元組長度，確保不會因為中文字元導致超長
        # 這裡需要考慮副檔名的位元組長度，從總長度中扣除
        while len(truncated_filename.encode('utf-8')) > MAX_NAME_LENGTH_BYTES - len(file_extension.encode('utf-8')):
            truncated_filename = truncated_filename[:-1]
        
        final_filename = truncated_filename + file_extension
        logger.warning(f"檔案名稱 '{base_name}' 超過 {MAX_NAME_LENGTH_CHARS} 字元或 {MAX_NAME_LENGTH_BYTES} 位元組，已截斷為 '{final_filename}'")
        return final_filename
    
    return base_name

CHECKPOINT_ENABLED = False
PASS = False
def checkpoint(label: str, data: dict = None):
    if not CHECKPOINT_ENABLED:
        return
    import inspect
    caller = inspect.stack()[1].function  # 自動顯示呼叫者函式名
    print(f"\nRUN: [{caller}] {label}")
    if data:
        for k, v in data.items():
            print(f"  {k}: {v}")
    if PASS:
        input(">>> 按 Enter 繼續...")

def process_string(input_string: str) -> str:
    """
    處理輸入的字串：
    1. 將所有英文字母大寫轉換為小寫。
    2. 將所有空格 ( ) 轉換為底線 (_)。

    Args:
        input_string: 需要處理的原始字串。

    Returns:
        處理後的字串，例如 "This is an example" 變成 "this_is_an_example"。
    """
    
    # 1. 將整個字串轉換為小寫
    lower_cased_string = input_string.lower()
    
    # 2. 使用 replace() 方法將所有單一空格替換為底線
    final_string = lower_cased_string.replace(" ", "_")
    
    return final_string

def clean_string(text: str) -> str:
    """
    清理字串，移除標點符號和符號，並轉換為小寫。
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s]', '', text) # 移除標點符號和符號
    return text.lower().strip()

# ================= 1. 全域變數 =================
CHECKPOINT_ENABLED = False
PASS = False

# ================= 2. 核心檢查類別 =================
class Checkpoint:
    def __init__(self, label: str):
        self.label = label

    # 「進門」要做的事 (with 開始時執行)
    def __enter__(self):
        if not CHECKPOINT_ENABLED:
            return None  # 回傳 None，後面的 if cp 就會跳過
        
        # 取得是誰在呼叫 (這行很直觀，就是找上一層的函式名)
        self.caller = inspect.stack()[1].function
        print(f"\n{'='*20} [CHECKPOINT: {self.caller}] {'='*20}")
        print(f"Label: {self.label}")
        return self  # 回傳自己，這樣才能用 cp.show()

    # 「出門」要做的事 (with 結束時執行)
    def __exit__(self, exc_type, exc_val, exc_tb):
        if CHECKPOINT_ENABLED and PASS:
            input("\n>>> [PAUSED] 按 Enter 繼續...")
            print(f"{'='*50}\n")

    # 功能函式：漂亮的印出資料
    def show(self, name: str, data):
        prefix = "    "
        print(f"\n  [Data] {name}:")
        
        # 1. 處理字典 (因為字典轉 list 只會剩下 Key，所以要分開處理)
        if isinstance(data, dict):
            items = list(data.items())
            print(f"{prefix}(Type: Dict, Keys: {len(data)})")
            for i, (k, v) in enumerate(items[:3]):
                print(f"{prefix}[{i}] {k}: {v} [Type: {type(v).__name__}]")
            if len(items) > 3: print(f"{prefix}...")
            return # 處理完字典直接結束

        # 2. 處理表格型資料 (Pandas/Numpy) 與 一般列表
        # 我們直接嘗試拿前 3 筆，不論它是什麼型態
        try:
            # 如果是 Pandas，它有 head() 最好用
            if hasattr(data, 'head'):
                print(f"{prefix}(Type: {type(data).__name__}, Shape: {getattr(data, 'shape', 'N/A')})")
                print(data.head(3))
            
            # 如果是 Numpy 或一般 List，我們直接用切片 [0:3]
            else:
                # 轉成 list 方便統一印出格式
                preview = list(data) 
                print(f"{prefix}(Type: {type(data).__name__}, Total: {len(preview)})")
                for i, item in enumerate(preview[:3]):
                    print(f"{prefix}[{i}] {item} [Type: {type(item).__name__}]")
                if len(preview) > 3: print(f"{prefix}...")
                
        except Exception:
            # 如果上面的嘗試都失敗了（例如資料不能被轉成 list），就直接嘗試切片硬印
            preview_data = data
            try:
                preview_data = data[0:3]
            except:
                pass
            print(f"{prefix}Value: {preview_data} [Type: {type(data).__name__}]")

# ================= 3. 主模組初始化函式 =================
def init_checkpoint(enabled: bool, pause: bool):
    global CHECKPOINT_ENABLED, PASS
    CHECKPOINT_ENABLED = enabled
    PASS = pause