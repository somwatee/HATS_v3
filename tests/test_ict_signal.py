import pandas as pd
import numpy as np
from src.ict_signal import (
    detect_swing_points,
    detect_mss,
    compute_fvg,
    compute_fibonacci_levels,
    generate_ict_signal
)

def build_dummy_df(n=100):
    """
    สร้าง DataFrame ตัวอย่างที่มีคอลัมน์จำเป็นทั้งหมด:
    time, open, high, low, close, tick_volume, atr, vwap, ema50_h4, ema200_h4, rsi_h4
    จากนั้นบังคับให้เกิด swing, MSS, FVG ในตำแหน่งที่กำหนด
    """
    times = pd.date_range("2025-01-01 07:00", periods=n, freq="T")
    df = pd.DataFrame({
        "time": times,
        "open": np.linspace(100, 110, n),
        "high": np.linspace(100.5, 110.5, n),
        "low": np.linspace(99.5, 109.5, n),
        "close": np.linspace(100.2, 110.2, n),
        "tick_volume": np.random.randint(50, 200, size=n),
        "atr": [0.2] * n,
        "vwap": np.linspace(100.2, 110.2, n),      # ทำให้ VWAP ค่อยๆขึ้น
        "ema50_h4": [105] * n,
        "ema200_h4": [100] * n,
        "rsi_h4": [55] * n
    })
    # บังคับให้มี swing low ที่ index=30, swing high ที่ index=40
    df.at[30, "low"] = 98
    df.at[40, "high"] = 112
    # MSS (bullish) ที่ index=41 (close > last swing high=112)
    df.at[41, "close"] = 113
    # บังคับ FVG (bullish) ที่ index=42:
    # prev1 (idx=41) low = df.at[41,'low'] = ปกติ ~ 100 → set ให้ low_prev1 > high_prev2
    df.at[40, "low"] = 100
    df.at[41, "low"] = 113  # low_prev1
    df.at[40, "high"] = 110  # high_prev2
    df.at[42, "open"] = 111
    df.at[42, "close"] = 112
    df.at[42, "high"] = 112.5
    return df

def test_detect_swing_points_and_mss():
    df = build_dummy_df()
    df_sw = detect_swing_points(df, window=5)
    assert df_sw.at[30, "is_swing_low"] == True
    assert df_sw.at[40, "is_swing_high"] == True

    df_mss = detect_mss(df_sw)
    assert df_mss.at[41, "bullish_mss"] == True
    assert df_mss.at[41, "bearish_mss"] == False

def test_compute_fvg_and_fibo():
    df0 = build_dummy_df()
    df1 = detect_swing_points(df0, window=5)
    df2 = detect_mss(df1)
    df3 = compute_fvg(df2)
    # index=42 ถูกกำหนดให้เป็น bullish_fvg
    assert df3.at[42, "bullish_fvg"] == True
    top = df3.at[42, "fvg_top"]
    bottom = df3.at[42, "fvg_bottom"]
    swing_low = df3.at[30, "low"]
    swing_high = df3.at[40, "high"]
    fibs = compute_fibonacci_levels(swing_low, swing_high)
    # ตรวจว่า fvg_top อยู่ระหว่าง fib_50 และ fib_382
    assert fibs["fib_382"] <= top <= fibs["fib_50"]

def test_generate_ict_signal():
    df0 = build_dummy_df(200)
    df1 = detect_swing_points(df0, window=5)
    df2 = detect_mss(df1)
    df3 = compute_fvg(df2)
    # idx ที่คาดว่าจะ generate สัญญาณคือ 42
    sig = generate_ict_signal(df3, idx=42)
    assert sig is not None
    assert sig["side"] == "Buy"
    assert "entry_price" in sig and "sl" in sig and "tp1" in sig
