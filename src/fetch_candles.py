import pandas as pd
from pathlib import Path
import MetaTrader5 as mt5
import yaml

# 1) โหลด config จากไฟล์ config/config.yaml
_cfg_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
with open(_cfg_path, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

# 2) อ่านค่าจาก config
symbol = _cfg["symbol"]
timeframe = _cfg["timeframe"]  # เช่น "M1"
n_bars = _cfg["fetch_candles_n"]
hist_path = Path(_cfg["historical_data_path"])

# 3) แปลงชื่อ timeframe ให้เป็น constant ของ MT5
TF_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1
}
TF = TF_MAP.get(timeframe, mt5.TIMEFRAME_M1)

def fetch_candles(n: int = n_bars) -> pd.DataFrame:
    """
    เชื่อม MT5, ดึง n แท่งเทียนล่าสุดสำหรับ symbol ตาม timeframe
    แล้วคืนค่าเป็น DataFrame ที่มีคอลัมน์:
    time, open, high, low, close, tick_volume
    """
    # 1. Initialize MT5
    if not mt5.initialize(
        path=_cfg["mt5"]["terminal_path"],
        login=_cfg["mt5"]["login"],
        server=_cfg["mt5"]["server"],
        password=_cfg["mt5"]["password"],
        timeout=_cfg["mt5"]["timeout"],
    ):
        print("MT5 Initialize failed")
        return pd.DataFrame()

    # 2. ดึงข้อมูลแท่งเทียนจาก MT5
    rates = mt5.copy_rates_from_pos(symbol, TF, 0, n)
    # 3. ปิดการเชื่อมต่อ MT5
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        print("No data retrieved from MT5")
        return pd.DataFrame()

    # 4. แปลงเป็น DataFrame และจัดรูปแบบ
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[["time", "open", "high", "low", "close", "tick_volume"]]
    return df

if __name__ == "__main__":
    # เมื่อรันเป็นสคริปต์หลัก จะดึง n_bars และบันทึกลงไฟล์ data/historical.csv
    df = fetch_candles()
    if df.empty:
        print("Fetched DataFrame is empty.")
    else:
        # สร้างโฟลเดอร์หากยังไม่มี
        Path(hist_path).parent.mkdir(parents=True, exist_ok=True)

        # ถ้าไฟล์ historical.csv มีอยู่เดิม ให้นำมาเชื่อมต่อ (concatenate) แล้ว drop duplicates
        if Path(hist_path).exists():
            try:
                df_old = pd.read_csv(hist_path, parse_dates=["time"])
                df = pd.concat([df_old, df]).drop_duplicates(subset="time").sort_values("time")
            except Exception as e:
                print(f"Warning: ไม่สามารถอ่านไฟล์เก่าได้: {e}")
        # บันทึกลง historical.csv
        df.to_csv(hist_path, index=False)
        print(f"Saved {len(df)} bars to {hist_path}")
