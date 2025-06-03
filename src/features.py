import pandas as pd
import numpy as np
import talib
import yaml
from pathlib import Path

# 1) โหลด config จาก config/config.yaml
_cfg_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
with open(_cfg_path, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

# 2) พาธ Input/Output จาก config
hist_path = Path(_cfg["historical_data_path"])      # data/historical.csv
feat_path = Path(_cfg["features_data_path"])         # data/data_with_features.csv

def compute_features(input_path: str, output_path: str):
    """
    อ่านไฟล์ historical.csv → คำนวณฟีเจอร์ทั้งหมด → บันทึกเป็น data_with_features.csv
    ฟีเจอร์:
      - ATR (14)
      - VWAP (สะสม)
      - EMA9, EMA21
      - RSI (14)
      - EMA50_H4, EMA200_H4, RSI_H4 (จากการ resample เป็น H4)
      - Bollinger Bands (period=20, stddev=2) + ผลต่างราคา–BB
      - MSS placeholder (False)
      - FVG placeholder (False)
      - ATR_MA (rolling 14)
      - Volume Imbalance = (close − open) / tick_volume
    """
    # โหลด historical prices
    df = pd.read_csv(input_path, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # 1) ATR M1 (period=14)
    df["atr"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)

    # 2) VWAP M1 (สะสม)
    cum_vp = (df["close"] * df["tick_volume"]).cumsum()
    cum_vol = df["tick_volume"].cumsum().replace(0, np.nan)
    df["vwap"] = cum_vp / cum_vol

    # 3) EMA9, EMA21 (บน M1)
    df["ema9"] = talib.EMA(df["close"], timeperiod=9)
    df["ema21"] = talib.EMA(df["close"], timeperiod=21)

    # 4) RSI14 (บน M1)
    df["rsi"] = talib.RSI(df["close"], timeperiod=14)

    # 5) MSS placeholder (ตั้งค่า False ก่อน)
    df["mss_bullish"] = False
    df["mss_bearish"] = False

    # 6) FVG placeholder (False)
    df["fvg_bullish"] = False
    df["fvg_bearish"] = False
    df["fvg_top"] = np.nan
    df["fvg_bottom"] = np.nan

    # 7) Higher-Timeframe H4 → คำนวณ EMA50_H4, EMA200_H4, RSI_H4
    df_h4 = df.set_index("time").resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "tick_volume": "sum"
    }).dropna().reset_index()
    df_h4["ema50_h4"] = talib.EMA(df_h4["close"], timeperiod=50)
    df_h4["ema200_h4"] = talib.EMA(df_h4["close"], timeperiod=200)
    df_h4["rsi_h4"] = talib.RSI(df_h4["close"], timeperiod=14)

    # Merge ค่า H4 ลง df หลัก แล้ว ffill
    df = df.merge(
        df_h4[["time", "ema50_h4", "ema200_h4", "rsi_h4"]],
        on="time",
        how="left"
    )
    df[["ema50_h4", "ema200_h4", "rsi_h4"]] = df[["ema50_h4", "ema200_h4", "rsi_h4"]].ffill()

    # 8) Bollinger Bands (period=20, stddev=2)
    upper, mid, lower = talib.BBANDS(df["close"], timeperiod=20, nbdevup=2, nbdevdn=2)
    df["bb_upper"] = upper
    df["bb_lower"] = lower

    # 9) ATR_MA (rolling 14 ของ ATR)
    df["atr_ma"] = df["atr"].rolling(window=14, min_periods=1).mean()

    # 10) คอลัมน์ diff กับ Bollinger
    df["bb_upper_diff"] = df["close"] - df["bb_upper"]
    df["bb_lower_diff"] = df["close"] - df["bb_lower"]

    # 11) Volume Imbalance = (close − open) / tick_volume
    df["vol_imbalance"] = (df["close"] - df["open"]) / df["tick_volume"].replace(0, np.nan)

    # 12) บันทึกไฟล์ features (สร้างโฟลเดอร์ output ถ้ายังไม่มี)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")

# เมื่อรันไฟล์นี้เป็นสคริปต์หลัก
if __name__ == "__main__":
    compute_features(str(hist_path), str(feat_path))
