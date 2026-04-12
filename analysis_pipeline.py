import pandas as pd
from typing import Tuple
from logs_handle import logger

# TODO: 需要建立完整的資料清理，分析工作流
def analyze_and_clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    對 DataFrame 進行分析，將資料內容為空的資料刪除，並建立前處理資料的 metadata
    （例如：清理規則、刪除的行數、資料類型轉換等）。
    目前此功能主要針對統計資料。
    """
    initial_rows = df.shape[0]
    cleaned_df = df.dropna(how='all') # 刪除所有欄位皆為空的行
    rows_deleted = initial_rows - cleaned_df.shape[0]

    # 這裡可以根據實際資料情況添加更多清理邏輯，例如：
    # - 處理特定欄位的缺失值
    # - 資料類型轉換 (例如：將數字字串轉換為數值型態)
    # - 移除重複行
    # - 處理異常值

    preprocessed_metadata = {
        "cleaning_rules": "刪除所有欄位皆為空的行",
        "rows_deleted": rows_deleted,
        "data_types_before_cleaning": {col: str(df[col].dtype) for col in df.columns},
        "data_types_after_cleaning": {col: str(cleaned_df[col].dtype) for col in cleaned_df.columns}
    }

    logger.notice(f"資料清理完成。原始行數: {initial_rows}, 清理後行數: {cleaned_df.shape[0]}, 刪除行數: {rows_deleted}")

    return cleaned_df, preprocessed_metadata

# TODO: Add preprocess_data_pipeline function


