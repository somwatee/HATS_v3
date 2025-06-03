import pandas as pd
import os
from pathlib import Path
import xgboost as xgb
import pytest
from src.model_trainer import train_walkforward

def create_dummy_dataset(tmp_path):
    """
    สร้างไฟล์ CSV ชั่วคราวที่มีคอลัมน์ฟีเจอร์และ label_ict
    โดยมีตัวอย่าง 20 แถว ให้มี Buy/Sell/NoTrade อย่างละบางส่วน
    """
    n = 20
    data = {
        "time": pd.date_range("2025-01-01", periods=n, freq="T"),
        "open": [i for i in range(n)],
        "high": [i + 0.5 for i in range(n)],
        "low": [i - 0.5 for i in range(n)],
        "close": [i + 0.2 for i in range(n)],
        "tick_volume": [100]*n,
        "atr": [0.1 + 0.01*i for i in range(n)],
        "vwap": [100 + 0.5*i for i in range(n)],
        "ema9": [50 + 0.2*i for i in range(n)],
        "ema21": [50 + 0.3*i for i in range(n)],
        "rsi": [30 + i for i in range(n)],
        "ema50_h4": [100]*n,
        "ema200_h4": [90]*n,
        "rsi_h4": [55]*n,
        "bb_upper": [110]*n,
        "bb_lower": [90]*n,
        "atr_ma": [0.15]*n,
        "bb_upper_diff": [i * 0.1 for i in range(n)],
        "bb_lower_diff": [-i * 0.1 for i in range(n)],
        "vol_imbalance": [0.1]*n,
        "mss_bullish": [0]*n,
        "mss_bearish": [0]*n,
        "fvg_bullish": [0]*n,
        "fvg_bearish": [0]*n,
    }

    # กระจาย label: แถว 5–7 เป็น Buy, 10–12 เป็น Sell, ที่เหลือ NoTrade
    labels = ["NoTrade"] * n
    labels[5] = labels[6] = labels[7] = "Buy"
    labels[10] = labels[11] = labels[12] = "Sell"
    data["label"] = labels

    df = pd.DataFrame(data)
    file_path = tmp_path / "with_labels_ict.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)

def test_train_walkforward_creates_model_and_report(tmp_path):
    # 1) สร้าง CSV ชั่วคราว
    dataset_file = create_dummy_dataset(tmp_path)

    # 2) กำหนดพาธไฟล์สำหรับโมเดลและรายงาน
    model_file = tmp_path / "xgb_test_model.json"
    report_file = tmp_path / "walkforward_test_report.txt"

    # 3) เรียก train_walkforward
    train_walkforward(dataset_file, str(model_file), str(report_file))

    # 4) ตรวจว่าไฟล์โมเดลและรายงานถูกสร้าง
    assert os.path.exists(str(model_file)), "Model file was not created"
    assert os.path.exists(str(report_file)), "Report file was not created"

    # 5) ตรวจว่าโมเดลโหลดได้ (ลอง load XGBClassifier)
    clf = xgb.XGBClassifier()
    clf.load_model(str(model_file))

    # 6) ตรวจว่ารายงานไม่ว่าง
    with open(str(report_file), "r", encoding="utf-8") as f:
        content = f.read()
    assert "Fold" in content or len(content) > 0

if __name__ == "__main__":
    pytest.main([__file__])
