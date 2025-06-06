import pandas as pd
import os
from pathlib import Path
from src.features import compute_features

def test_compute_features(tmp_path):
    """
    สร้าง DataFrame ขนาดเล็ก 10 แท่ง → บันทึกเป็น CSV ชั่วคราว → เรียก compute_features
    ตรวจว่าไฟล์ output ถูกสร้าง และมีคอลัมน์ฟีเจอร์หลักครบ
    """
    # 1. สร้าง DataFrame ตัวอย่าง
    data = {
        "time": pd.date_range("2025-01-01 00:00", periods=10, freq="T"),
        "open": [1,2,3,4,5,6,7,8,9,10],
        "high": [2,3,4,5,6,7,8,9,10,11],
        "low": [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5],
        "close": [1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5],
        "tick_volume": [100]*10
    }
    df = pd.DataFrame(data)
    input_file = tmp_path / "hist.csv"
    df.to_csv(input_file, index=False)

    # 2. เรียก compute_features → บันทึกไฟล์ output
    output_file = tmp_path / "feat.csv"
    compute_features(str(input_file), str(output_file))

    # 3. ตรวจว่าไฟล์ถูกสร้าง
    assert os.path.exists(str(output_file))

    # 4. โหลดผลลัพธ์ และเช็คคอลัมน์ฟีเจอร์หลัก
    df_out = pd.read_csv(output_file)
    expected_cols = [
        "time", "open", "high", "low", "close", "tick_volume",
        "atr", "vwap",
        "ema9", "ema21", "rsi",
        "mss_bullish", "mss_bearish",
        "fvg_bullish", "fvg_bearish", "fvg_top", "fvg_bottom",
        "ema50_h4", "ema200_h4", "rsi_h4",
        "bb_upper", "bb_lower", "atr_ma",
        "bb_upper_diff", "bb_lower_diff",
        "vol_imbalance"
    ]
    for col in expected_cols:
        assert col in df_out.columns
