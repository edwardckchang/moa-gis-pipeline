from logs_handle import logger

def handle_save_menu(file_path: str, action_description: str, save_action_description: str, name: str = "") -> bool:
    """
    處理資料儲存的選單邏輯，提供使用者選擇是否儲存或合併資料。

    Args:
        file_path (str): 儲存檔案的路徑。
        action_description (str): 描述資料已完成的動作 (例如 "排序", "處理")。
        save_action_description (str): 描述儲存動作的名稱 (例如 "合併", "儲存")。
        name (str): 可提供的檔案名稱。預設為空字串。
    Returns:
        bool: 如果資料成功儲存則返回 True，否則返回 False。
    """
    if not file_path:
        logger.error("未提供檔案路徑，無法執行儲存操作。")
        return False
    if name:
        question = f" '{name}' 的資料已{action_description}，是否{save_action_description}?"
    else:
        question = f"資料已{action_description}，是否{save_action_description}?"
    if yes_no_menu(question):
        logger.notice(f"將{name}{action_description}後的資料儲存到 '{file_path}'。")
        return True
    else:
        logger.info(f"使用者取消儲存資料到檔案 '{file_path}' 。")
        return False
    
AUTO_CONFIRMED = False
AUTO_YES = False
AUTO_NO = False

def make_AUTO_YES():
    global AUTO_CONFIRMED
    AUTO_CONFIRMED = True
    global AUTO_YES
    AUTO_YES = True
    global AUTO_NO
    AUTO_NO = False

def make_AUTO_NO():
    global AUTO_CONFIRMED
    AUTO_CONFIRMED = True
    global AUTO_NO
    AUTO_NO = True
    global AUTO_YES
    AUTO_YES = False

def disable_auto_confirm():
    global AUTO_CONFIRMED
    AUTO_CONFIRMED = False
    global AUTO_YES
    AUTO_YES = False
    global AUTO_NO
    AUTO_NO = False

def yes_no_menu(question: str) -> bool:
    global AUTO_CONFIRMED, AUTO_YES, AUTO_NO
    if AUTO_CONFIRMED:
        return True if AUTO_YES else False if AUTO_NO else None
    while True:
        answer = input(question + " (y/n): ")
        if answer.lower() == "y":
            return True
        elif answer.lower() == "n":
            return False
