import pandas as pd
import os
from pathlib import Path
from src.build_labels import build_labels

def test_build_labels(tmp_path):
    """
    สร้าง DataFrame ฟีเจอร์ขนาดเล็ก แล้วทดสอบว่า build_labels() สร้างคอลัมน์ `label`
    และไฟล์ CSV ถูกสร้าง
    """
    # 1) สร้าง DataFrame ชั่วคราว (10 แท่ง) พร้อมฟีเจอร์พื้นฐาน (บางค่าอาจ NaN)
    data = {
        "time": pd.date_range("2025-01-01", periods=10, freq="T"),
        "open": [1,2,3,4,5,4,3,2,1,2],
        "high": [2,3,4,5,6,5,4,3,2,3],
        "low": [0.5,1.5,2.5,3.5,4.5,3.5,2.5,1.5,0.5,1.0],
        "close": [1.5,2.5,3.5,4.5,5.5,4.5,3.5,2.5,1.5,2.0],
        "tick_volume": [100]*10,
        # ฟีเจอร์ง่ายๆ (ไม่ต้องสมจริง) เพื่อทดสอบ flow แต่ละคอลัมน์ต้องมี
        "ema50_h4": [1]*10, "ema200_h4": [0.5]*10, "rsi_h4": [60]*10,
        "vwap": [1]*10, "atr": [0.1]*10, "bb_lower": [1]*10, "bb_upper": [5]*10,
        "atr_ma": [0.2]*10, "rsi": [25]*10, "adx": [30]*10,
        "mss_bullish": [False]*10, "mss_bearish": [False]*10,
        "fvg_bullish": [False]*10, "fvg_bearish": [False]*10,
        "fib_in_zone": [False]*10, "vol_imbalance": [0.1]*10
    }
    df = pd.DataFrame(data)
    input_file = tmp_path / "features.csv"
    df.to_csv(input_file, index=False)

    # 2) เรียก build_labels → ผลลัพธ์ควรสร้างไฟล์ใหม่
    output_file = tmp_path / "labels.csv"
    build_labels(str(input_file), str(output_file))

    # 3) ตรวจไฟล์ถูกสร้าง
    assert os.path.exists(str(output_file))

    # 4) โหลดผลลัพธ์และตรวจว่ามีคอลัมน์ "label"
    df_out = pd.read_csv(output_file)
    assert "label" in df_out.columns
    # ค่า label ควรเป็น str ทั้งหมด
    assert all(isinstance(x, str) for x in df_out["label"])
