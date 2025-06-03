import pandas as pd
import numpy as np
import xgboost as xgb
import yaml
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit

# โหลด config
_cfg_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
with open(_cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# คอลัมน์ฟีเจอร์ที่จะใช้ (ต้องมีใน with_labels_ict.csv)
FEATURE_COLS = [
    "atr", "vwap",
    "ema9", "ema21", "rsi",
    "ema50_h4", "ema200_h4", "rsi_h4",
    "bb_upper", "bb_lower", "atr_ma",
    "bb_upper_diff", "bb_lower_diff",
    "vol_imbalance",
    "mss_bullish", "mss_bearish",
    "fvg_bullish", "fvg_bearish"
]

# พารามิเตอร์ XGBoost (อ่านจาก config ถ้ามี)
params = {
    "objective":       "multi:softprob",
    "num_class":       3,
    "eval_metric":     "mlogloss",
    "max_depth":       cfg.get("xgb_max_depth", 4),
    "eta":             cfg.get("xgb_eta", 0.05),
    "subsample":       cfg.get("xgb_subsample", 0.8),
    "colsample_bytree":cfg.get("xgb_colsample_bytree", 0.8),
    "random_state":    42,
}

def train_walkforward(dataset_path: str,
                      model_output: str,
                      report_output: str):
    """
    อ่านไฟล์ CSV (with_labels_ict.csv) → แยก X, y → Walk-forward CV → บันทึกรายงาน + สร้างโมเดลสุดท้าย
    Args:
      dataset_path: พาธไปยัง data/with_labels_ict.csv
      model_output:  พาธที่จะบันทึกไฟล์โมเดล XGBoost (.json)
      report_output: พาธที่จะบันทึกรายงาน walk-forward (.txt)
    """
    df = pd.read_csv(dataset_path)

    # แปลง boolean เป็น int (0/1)
    for col in ["mss_bullish", "mss_bearish", "fvg_bullish", "fvg_bearish"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # เตรียมคุณลักษณะ X
    X = df[FEATURE_COLS]

    # แปลง label เป็นตัวเลข: Buy→1, Sell→2, NoTrade→0
    df["label_num"] = df["label"].map({"Buy": 1, "Sell": 2, "NoTrade": 0})
    y = df["label_num"]

    # Walk-forward CV
    tscv = TimeSeriesSplit(n_splits=cfg.get("walkforward_splits", 5))
    fold = 0
    reports = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf = xgb.XGBClassifier(**params)
        clf.fit(X_train, y_train)

        # พยายามทำนาย class labels 1D
        y_pred_raw = clf.predict(X_test)
        if isinstance(y_pred_raw, np.ndarray) and y_pred_raw.ndim > 1:
            y_pred = np.argmax(y_pred_raw, axis=1)
        else:
            y_pred = y_pred_raw

        report = classification_report(
            y_test, y_pred, labels=[1, 2, 0],
            target_names=["Buy", "Sell", "NoTrade"]
        )
        header = f"=== Fold {fold} ===\n"
        reports.append(header + report + "\n")
        fold += 1

    # บันทึกรายงาน walk-forward
    Path(report_output).parent.mkdir(parents=True, exist_ok=True)
    with open(report_output, "w", encoding="utf-8") as f:
        f.writelines(reports)

    # เทรนโมเดลบนข้อมูลทั้งหมด แล้วบันทึกโมเดล
    clf_final = xgb.XGBClassifier(**params)
    clf_final.fit(X, y)
    Path(model_output).parent.mkdir(parents=True, exist_ok=True)
    clf_final.save_model(model_output)
    print(f"Model saved to {model_output}")


if __name__ == "__main__":
    train_walkforward(
        cfg["dataset_path"],
        cfg["model_path"],
        str(Path(cfg["model_path"]).parent / "walkforward_report.txt")
    )
