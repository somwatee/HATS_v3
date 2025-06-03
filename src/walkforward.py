import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict

def run_walkforward(
    X: pd.DataFrame,
    y: pd.Series,
    params: Dict,
    n_splits: int = 5
) -> Tuple[List[str], xgb.XGBClassifier]:
    """
    ทำ Walk‐forward Cross‐Validation ด้วย TimeSeriesSplit

    Args:
      X: pandas.DataFrame ของ feature variables
      y: pandas.Series ของ label (ตัวเลข 0/1/2 ฯลฯ)
      params: พารามิเตอร์สำหรับ XGBClassifier
      n_splits: จำนวน fold สำหรับ TimeSeriesSplit

    Returns:
      reports: List[str] รายงาน classification_report ของแต่ละ fold
      final_model: XGBClassifier ที่เทรนบนข้อมูลทั้งหมดแล้ว
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    reports: List[str] = []

    unique_labels = np.unique(y)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf = xgb.XGBClassifier(**params)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        report = classification_report(
            y_test,
            y_pred,
            labels=unique_labels.tolist(),
            target_names=[str(label) for label in unique_labels]
        )
        header = f"=== Fold {fold} ===\n"
        reports.append(header + report + "\n")

    # เทรนโมเดลบนข้อมูลทั้งหมด
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y)

    return reports, final_model
