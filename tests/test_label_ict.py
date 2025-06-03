import pandas as pd
import numpy as np
from pathlib import Path
import os
from src.label_ict import label_ict

def build_dummy_features(tmp_path):
    """
    สร้าง DataFrame ฟีเจอร์เบื้องต้น (มี col ที่ generate_ict_signal ต้องการ)
    เช่น time, open, high, low, close, tick_volume, atr, vwap, ema50_h4, ema200_h4, rsi_h4,
         (นอกจากนี้ detect_swing_points, detect_mss, compute_fvg จะสร้าง col ใหม่เอง)
    """
    n = 100
    times = pd.date_range("2025-01-01 07:00", periods=n, freq="T")
    df = pd.DataFrame({
        "time": times,
        "open": np.linspace(100, 110, n),
        "high": np.linspace(100.5, 110.5, n),
        "low": np.linspace(99.5, 109.5, n),
        "close": np.linspace(100.2, 110.2, n),
        "tick_volume": np.random.randint(50, 200, size=n),
        "atr": [0.2] * n,
        "vwap": (np.linspace(100.2, 110.2, n) * np.random.randint(50, 200, size=n)).cumsum() /
                np.random.randint(50, 200, size=n).cumsum(),
        "ema50_h4": [105] * n,
        "ema200_h4": [100] * n,
        "rsi_h4": [55] * n
    })
    # บังคับให้มี swing & MSS & FVG บางจุด
    # เช่น index=50 เป็น swing high, index=40 เป็น swing low
    df.at[40, "low"] = 98
    df.at[50, "high"] = 112
    # MSS ที่ index=51 (close > swing_high)
    df.at[51, "close"] = 113
    # FVG ที่ index=52
    df.at[50, "low"] = 100
    df.at[51, "low"] = 111.5
    df.at[51, "high"] = 112.2
    df.at[52, "close"] = 113
    df.at[52, "open"] = 112
    df.at[52, "high"] = 113.5
    return df

def test_label_ict(tmp_path):
    # 1) สร้างไฟล์ชั่วคราว .csv ลงใน tmp_path
    df_feat = build_dummy_features(tmp_path)
    input_file = tmp_path / "features.csv"
    df_feat.to_csv(input_file, index=False)

    # 2) เรียก label_ict → บันทึกไฟล์ output
    output_file = tmp_path / "labels.csv"
    label_ict(str(input_file), str(output_file))

    # 3) ตรวจว่าไฟล์ถูกสร้าง
    assert os.path.exists(str(output_file))

    # 4) โหลดผลลัพธ์ เช็คว่ามีคอลัมน์ label และมีค่า “Buy” อยู่อย่างน้อยหนึ่งแถว
    df_out = pd.read_csv(output_file)
    assert "label" in df_out.columns
    assert "Buy" in df_out["label"].values or "Sell" in df_out["label"].values
