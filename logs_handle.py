import logging
import os
from datetime import datetime

# 定義自定義日誌級別
SUCCESS_LEVEL_NUM = 25
NOTICE_LEVEL_NUM = 15
EXECUTION_LEVEL_NUM = 27
LOGS_LEVEL_NUM = 26
logging.addLevelName(SUCCESS_LEVEL_NUM, 'SUCCESS')
logging.addLevelName(NOTICE_LEVEL_NUM, 'NOTICE')
logging.addLevelName(EXECUTION_LEVEL_NUM, 'EXECUTION')
logging.addLevelName(LOGS_LEVEL_NUM, 'LOGS')

# 添加自定義方法
def success(self, message, *args, **kwargs):
    if self.isEnabledFor(SUCCESS_LEVEL_NUM):
        self._log(SUCCESS_LEVEL_NUM, message, args, **kwargs)

def notice(self, message, *args, **kwargs):
    if self.isEnabledFor(NOTICE_LEVEL_NUM):
        self._log(NOTICE_LEVEL_NUM, message, args, **kwargs)

def execution(self, message, *args, **kwargs):
    if self.isEnabledFor(EXECUTION_LEVEL_NUM):
        self._log(EXECUTION_LEVEL_NUM, message, args, **kwargs)

def logs(self, message, *args, **kwargs):
    if self.isEnabledFor(LOGS_LEVEL_NUM):
        self._log(LOGS_LEVEL_NUM, message, args, **kwargs)

logging.Logger.execution = execution
logging.Logger.success = success
logging.Logger.notice = notice
logging.Logger.logs = logs

# 自定義過濾器
class SuccessFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == SUCCESS_LEVEL_NUM

class ExecutionFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == EXECUTION_LEVEL_NUM

class LogsFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == LOGS_LEVEL_NUM

class ErrorFilter(logging.Filter):
    def filter(self, record):
        return record.levelno in (logging.ERROR, logging.CRITICAL)

# 配置日誌
logger = logging.getLogger(__name__)

# 自定義條件格式化器
class ConditionalFormatter(logging.Formatter):
    def __init__(self, datefmt=None, style='%'):
        super().__init__(datefmt=datefmt, style=style)
        self.simple_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt=datefmt, style=style)
        self.detailed_format = logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s', datefmt=datefmt, style=style)

    def format(self, record):
        if record.levelno == 10:
            return self.detailed_format.format(record)
        if record.levelno <= SUCCESS_LEVEL_NUM:
            return self.simple_format.format(record)
        else:
            return self.detailed_format.format(record)

class LazyFileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None):
        self.baseFilename = os.path.abspath(filename)
        self.mode = mode
        self.encoding = encoding
        self.errors = None
        self.stream = None
        logging.Handler.__init__(self)

    def emit(self, record):
        if self.stream is None:
            self.stream = self._open()
        super().emit(record)

def setup_logging(level=15, log_dir="logs"):
    """
    配置日誌系統，支援控制台和分級檔案輸出。

    Args:
        level (int): 日誌級別，預設從環境變數 LOG_LEVEL 或 SUCCESS_LEVEL_NUM。
        log_dir (str): 日誌檔案儲存資料夾，預設為 'logs'。
    """
    try:
        if level is None:
            level = int(os.getenv('LOG_LEVEL', SUCCESS_LEVEL_NUM))
        
        # 設置根日誌記錄器和所有現有日誌記錄器的級別
        logging.root.setLevel(level)
        for name, log in logging.Logger.manager.loggerDict.items():
            if isinstance(log, logging.Logger):
                log.setLevel(level)
                log.propagate = True
        logger.setLevel(level)
        logger.propagate = True
        logging.root.handlers.clear()

        conditional_formatter = ConditionalFormatter()

        # 將 log_dir 轉換為絕對路徑，確保其相對於腳本所在目錄
        # 獲取當前檔案 (logs_handle.py) 的目錄
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        # 將 log_dir 與當前檔案目錄結合，形成絕對路徑
        absolute_log_dir = os.path.join(current_file_dir, log_dir)

        if not os.path.exists(absolute_log_dir):
            os.makedirs(absolute_log_dir)
        
        today = datetime.now().strftime("%Y%m%d")
        # 使用 absolute_log_dir 來建構日誌檔案的絕對路徑
        success_log_file = os.path.join(absolute_log_dir, f"logs_success_{today}.log")
        execution_log_file = os.path.join(absolute_log_dir, f"logs_success_{today}.log") # 注意：這裡與 success_log_file 相同，是預期行為
        logs_log_file = os.path.join(absolute_log_dir, f"logs_logs_{today}.log")
        error_log_file = os.path.join(absolute_log_dir, f"logs_error_{today}.log")

        # 添加控制台處理器（NOTICE 將輸出到這裡）
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)  # 確保接受 DEBUG 以上的日誌
        console_handler.setFormatter(conditional_formatter)
        logging.root.addHandler(console_handler)

        # 添加 execution 日誌檔案處理器
        execution_handler = LazyFileHandler(execution_log_file, mode='a', encoding='utf-8')
        execution_handler.setLevel(EXECUTION_LEVEL_NUM)
        execution_handler.addFilter(ExecutionFilter())
        execution_handler.setFormatter(conditional_formatter)
        logging.root.addHandler(execution_handler)

        # 添加 SUCCESS 日誌檔案處理器
        success_handler = LazyFileHandler(success_log_file, mode='a', encoding='utf-8')
        success_handler.setLevel(SUCCESS_LEVEL_NUM)
        success_handler.addFilter(SuccessFilter())
        success_handler.setFormatter(conditional_formatter)
        logging.root.addHandler(success_handler)

        # 添加 LOGS 日誌檔案處理器
        logs_handler = LazyFileHandler(logs_log_file, mode='a', encoding='utf-8')
        logs_handler.setLevel(LOGS_LEVEL_NUM)
        logs_handler.addFilter(LogsFilter())
        logs_handler.setFormatter(conditional_formatter)
        logging.root.addHandler(logs_handler)

        # 添加 ERROR 和 CRITICAL 日誌檔案處理器
        error_handler = LazyFileHandler(error_log_file, mode='a', encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.addFilter(ErrorFilter())
        error_handler.setFormatter(conditional_formatter)
        logging.root.addHandler(error_handler)

        # 不再為 NOTICE 添加獨立的 FileHandler，NOTICE 日誌將通過 console_handler 輸出
        logger.debug(f"Root logger level: {logging.root.getEffectiveLevel()}")
        logger.debug(f"Module logger level: {logger.getEffectiveLevel()}")
        logger.notice("setup_logging 函數配置完成")

    except Exception as e:
        logger.error(f"日誌配置失敗: {str(e)}")
        raise

class TkinterTextHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    def emit(self, record):
        msg = self.format(record)
        self.text_widget.insert('end', msg + '\n')
        self.text_widget.see('end')

def setup_logging_to_tkinter(text_widget):
    """
    配置 logging 模組，使其將日誌輸出到 Tkinter 的文本區域。
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    handler = TkinterTextHandler(text_widget)
    handler.setLevel(logging.DEBUG)
    logging.root.addHandler(handler)
    logger.setLevel(logging.DEBUG)
