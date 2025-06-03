import pandas as pd
import numpy as np
import talib
import yaml
from pathlib import Path

# 1) โหลด config จาก config/config.yaml
_cfg_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
with open(_cfg_path, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

# 2) อ่านพาธจาก config
hist_path = Path(_cfg["historical_data_path"])
feat_path = Path(_cfg["features_data_path"])

def compute_features(input_path: str, output_path: str):
    """
    อ่านไฟล์ historical.csv → คำนวณฟีเจอร์ทั้งหมด → บันทึก CSV ใหม่ชื่อ data_with_features.csv
    """
    # 1. โหลด historical prices
    df = pd.read_csv(input_path, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # 2. MSS (Market Structure Shift) – ตัวอย่างง่าย
    df["mss_bullish"] = False
    df["mss_bearish"] = False
    for i in range(2, len(df) - 2):
        # MSS Bearish: high ลดลงติดกัน 2 แท่ง
        if df["high"].iloc[i] < df["high"].iloc[i - 1] and df["high"].iloc[i - 1] < df["high"].iloc[i - 2]:
            df.at[i, "mss_bearish"] = True
        # MSS Bullish: low ขึ้นติดกัน 2 แท่ง
        if df["low"].iloc[i] > df["low"].iloc[i - 1] and df["low"].iloc[i - 1] > df["low"].iloc[i - 2]:
            df.at[i, "mss_bullish"] = True

    # 3. FVG (Fair Value Gap) & fib_in_zone – placeholder (เริ่มต้นเซ็ต False)
    df["fvg_bullish"] = False
    df["fvg_bearish"] = False
    df["fib_in_zone"] = False
    # ※ สามารถเขียนกฎคำนวณ FVG และเช็ค Fibonacci zone ได้ในภายหลัง

    # 4. RSI, EMA9, EMA21 ด้วย TA-Lib
    df["rsi"] = talib.RSI(df["close"], timeperiod=14)
    df["ema9"] = talib.EMA(df["close"], timeperiod=9)
    df["ema21"] = talib.EMA(df["close"], timeperiod=21)

    # 5. ATR, ADX (Period 14)
    df["atr"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)
    df["adx"] = talib.ADX(df["high"], df["low"], df["close"], timeperiod=14)

    # 6. Volume Imbalance (ตัวอย่างง่าย: (close - open) / tick_volume)
    df["vol_imbalance"] = (df["close"] - df["open"]) / df["tick_volume"].replace(0, np.nan)

    # 7. VWAP & VWAP_DIFF
    df["vwap"] = (df["close"] * df["tick_volume"]).cumsum() / df["tick_volume"].cumsum()
    df["vwap_diff"] = df["open"] - df["vwap"]

    # 8. Higher-Timeframe Signals (resample M1 → H4)
    #    สร้าง DataFrame ย่อยสำหรับ H4
    df_h4 = df.set_index("time").resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "tick_volume": "sum"
    }).dropna().reset_index()

    # คำนวณ EMA50, EMA200, RSI บน H4
    df_h4["ema50_h4"] = talib.EMA(df_h4["close"], timeperiod=50)
    df_h4["ema200_h4"] = talib.EMA(df_h4["close"], timeperiod=200)
    df_h4["rsi_h4"] = talib.RSI(df_h4["close"], timeperiod=14)

    # รวมค่าจาก H4 ลงใน df หลัก (ใช้ ffill เติมค่า)
    df = df.merge(
        df_h4[["time", "ema50_h4", "ema200_h4", "rsi_h4"]],
        on="time",
        how="left"
    )
    df[["ema50_h4", "ema200_h4", "rsi_h4"]] = df[["ema50_h4", "ema200_h4", "rsi_h4"]].ffill()

    # 9. Bollinger Bands (period=20, stddev=2)
    upper, mid, lower = talib.BBANDS(df["close"], timeperiod=20, nbdevup=2, nbdevdn=2)
    df["bb_upper"] = upper
    df["bb_lower"] = lower

    # ค่าเฉลี่ย ATR (ATR_MA) บนช่วง 14 แท่ง
    df["atr_ma"] = df["atr"].rolling(window=14, min_periods=1).mean()

    # 10. คอลัมน์สรุป diff ราคากับ BB
    df["bb_upper_diff"] = df["close"] - df["bb_upper"]
    df["bb_lower_diff"] = df["close"] - df["bb_lower"]

    # 11. บันทึกไฟล์ features (สร้างโฟลเดอร์หากยังไม่มี)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")

# เมื่อรันเป็นสคริปต์หลัก
if __name__ == "__main__":
    compute_features(str(hist_path), str(feat_path))
