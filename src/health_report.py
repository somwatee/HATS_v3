import yaml
import requests
from pathlib import Path
import MetaTrader5 as mt5
from datetime import datetime

# โหลด config
_cfg_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
with open(_cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

TELE_BOT  = cfg["telegram"]["bot_token"]
TELE_CHAT = cfg["telegram"]["chat_id"]
ALERT_ON  = cfg["telegram"]["alert_on"]
MT5_CFG   = cfg["mt5"]

def send_telegram(message: str):
    """
    ส่งข้อความไปยัง Telegram ตาม bot_token และ chat_id ใน config
    """
    url = f"https://api.telegram.org/bot{TELE_BOT}/sendMessage"
    payload = {"chat_id": TELE_CHAT, "text": message}
    try:
        requests.post(url, data=payload, timeout=5)
    except Exception as e:
        print(f"[{datetime.now()}] Failed to send Telegram alert: {e}")

def check_mt5_connection() -> bool:
    """
    ตรวจสอบว่า MT5 Terminal เชื่อมต่อได้หรือไม่
    คืน True หากเชื่อมต่อสำเร็จ, False หากล้มเหลวหรือเกิดข้อผิดพลาด
    """
    try:
        if mt5.initialize(
            path=MT5_CFG["terminal_path"],
            login=MT5_CFG["login"],
            server=MT5_CFG["server"],
            password=MT5_CFG["password"],
            timeout=MT5_CFG["timeout"],
        ):
            mt5.shutdown()
            return True
    except Exception as e:
        print(f"[{datetime.now()}] Exception during MT5 initialize: {e}")
    return False

def health_check():
    """
    ตรวจเช็คสุขภาพระบบ:
    1) MT5 connection ถ้าเชื่อมไม่สำเร็จ & "connection_error" ใน alert_on → ส่ง Telegram
    2) ตรวจไฟล์ logs/system.log ถ้ามีคำว่า "ERROR" & "system_health" ใน alert_on → ส่ง Telegram
    """
    # 1) MT5 Connection
    try:
        ok_mt5 = check_mt5_connection()
        if not ok_mt5 and "connection_error" in ALERT_ON:
            send_telegram(f"[{datetime.now()}] ALERT: MT5 connection failed")
    except Exception as e:
        print(f"[{datetime.now()}] Exception in health_check MT5 check: {e}")

    # 2) System log errors
    log_path = Path("logs") / "system.log"
    if log_path.exists() and "system_health" in ALERT_ON:
        try:
            content = log_path.read_text(encoding="utf-8", errors="ignore")
            if "ERROR" in content:
                send_telegram(f"[{datetime.now()}] ALERT: 'ERROR' found in system.log")
        except Exception as e:
            print(f"[{datetime.now()}] Failed to read system.log: {e}")

if __name__ == "__main__":
    health_check()
