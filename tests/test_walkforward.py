import pandas as pd
import numpy as np
import os
from pathlib import Path
from src.walkforward import run_walkforward
import xgboost as xgb

def create_dummy_X_y(tmp_path):
    """
    สร้าง X (DataFrame) และ y (Series) ชั่วคราว
    ให้มีข้อมูลอย่างน้อย n_splits+1 แถว เพื่อทดสอบ walkforward
    """
    n = 20
    # สร้าง features สุ่ม
    data = {
        "f1": np.linspace(0, 1, n),
        "f2": np.linspace(1, 2, n),
        "f3": np.random.randn(n)
    }
    X = pd.DataFrame(data)

    # สร้าง labels สลับกัน 0,1,2
    labels = []
    for i in range(n):
        labels.append(i % 3)
    y = pd.Series(labels, name="label")

    return X, y

def test_run_walkforward_creates_reports_and_model(tmp_path):
    # 1) สร้าง X และ y dummy
    X, y = create_dummy_X_y(tmp_path)

    # 2) กำหนดพารามิเตอร์ XGBoost อย่างง่าย
    params = {
        "objective":   "multi:softprob",
        "num_class":   3,
        "eval_metric": "mlogloss",
        "max_depth":   2,
        "eta":         0.1,
        "random_state": 42
    }

    # 3) รัน walkforward
    n_splits = 4
    reports, model = run_walkforward(X, y, params, n_splits=n_splits)

    # 4) เช็คจำนวนรายงานถูกต้อง
    assert isinstance(reports, list)
    assert len(reports) == n_splits

    # 5) เช็คเนื้อหารายงานแต่ละ fold มีคำว่า "precision" (มาจาก classification_report)
    for rpt in reports:
        assert "precision" in rpt

    # 6) เช็คโมเดลที่ได้เป็น XGBClassifier และทำนายได้
    assert isinstance(model, xgb.XGBClassifier)
    y_pred_all = model.predict(X)
    assert len(y_pred_all) == len(y)
    # ทุกค่าทำนายต้องเป็น int ใน set {0,1,2}
    assert set(y_pred_all).issubset({0, 1, 2})

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
