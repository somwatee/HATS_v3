import pandas as pd
import os
import xgboost as xgb
import pytest
from pathlib import Path

# เราจะ monkeypatch yaml.safe_load ใน run_phase2 เพื่อชี้ไปยัง dataset + model path ชั่วคราว
import builtins
import yaml

from src.model_trainer import train_walkforward

# สร้าง dummy dataset สำหรับ Phase 2 (เหมือนใน test_model_trainer)
def create_dummy_dataset(tmp_path):
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
    labels = ["NoTrade"] * n
    labels[5] = labels[6] = labels[7] = "Buy"
    labels[10] = labels[11] = labels[12] = "Sell"
    data["label"] = labels

    df = pd.DataFrame(data)
    file_path = tmp_path / "with_labels_ict.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)

def test_run_phase2_creates_model_and_report(tmp_path, monkeypatch):
    # 1) สร้าง CSV เก็บใน tmp_path
    dataset_file = create_dummy_dataset(tmp_path)

    # 2) กำหนดพาธโมเดลและรายงานใน tmp_path
    model_file = tmp_path / "xgb_test_model.json"
    report_file = tmp_path / "walkforward_report.txt"

    # 3) โครงสร้าง config ชั่วคราว
    fake_cfg = {
        "dataset_path": dataset_file,
        "model_path": str(model_file),
        # อื่นๆ ไม่จำเป็นสำหรับ test
    }

    # 4) Monkeypatch yaml.safe_load ใน run_phase2 ให้คืน fake_cfg
    import scripts.run_phase2 as rp2
    monkeypatch.setattr(yaml, "safe_load", lambda f: fake_cfg)

    # 5) เรียก run_phase2.main()
    rp2.main()

    # 6) ตรวจว่ามีไฟล์โมเดลและรายงานถูกสร้าง
    assert os.path.exists(str(model_file)), "Model file was not created"
    assert os.path.exists(str(report_file)), "Report file was not created"

    # 7) ตรวจว่าโหลดโมเดลได้
    clf = xgb.XGBClassifier()
    clf.load_model(str(model_file))

    # 8) ตรวจว่ารายงานไม่ว่าง (มีอย่างน้อยคำว่า 'Fold')
    content = Path(report_file).read_text(encoding="utf-8")
    assert "Fold" in content

if __name__ == "__main__":
    pytest.main([__file__])
