import subprocess
import os
from datetime import datetime
from dotenv import load_dotenv
from logs_handle import logger

def run_manual_backup():
    load_dotenv()
    
    # 從 .env 讀取設定
    user = os.getenv("DB_USER")
    db = os.getenv("DB_NAME")
    password = os.getenv("DB_PASSWORD")
    
    date_str = datetime.now().strftime("%Y%m%d")
    backup_path = f"backups/moa_opendata/backup_{date_str}.sql"
    
    # 建立目錄
    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    
    # 設定環境變數供子程序使用
    env = os.environ.copy()
    env["PGPASSWORD"] = password
    
    cmd = [
        "pg_dump",
        "-U", user,
        "-d", db,
        "-f", backup_path
    ]
    
    try:
        subprocess.run(cmd, env=env, check=True)
        logger.info(f"資料庫備份成功：{backup_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"備份失敗：{e}")

if __name__ == "__main__":
    run_manual_backup()