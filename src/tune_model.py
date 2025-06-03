import pandas as pd
import yaml
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

# โหลด config
_cfg_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
with open(_cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# โหลด dataset จากไฟล์ ICT‐based labels
df = pd.read_csv(cfg["dataset_path"])  # data/with_labels_ict.csv

# ฟีเจอร์เดียวกันกับ model_trainer.py
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

# แปลง boolean เป็น int
for col in ["mss_bullish", "mss_bearish", "fvg_bullish", "fvg_bearish"]:
    if col in df.columns:
        df[col] = df[col].astype(int)

X = df[FEATURE_COLS]
y = df["label"].map({"Buy": 1, "Sell": 2, "NoTrade": 0})

# แบ่งข้อมูล train/test 80/20 (stratify ถ้าเป็นไปได้)
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

# กำหนด grid ของ hyperparameters
param_grid = {
    "max_depth":        [3, 4, 5],
    "eta":              [0.01, 0.05, 0.1],
    "subsample":        [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "scale_pos_weight": [1, 5, 10]
}

model = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42
)

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1
)

print("Starting GridSearchCV...")
grid.fit(X_train, y_train)

print("Best hyperparameters:", grid.best_params_)

# บันทึกผลลัพธ์ CV ทั้งหมด
results_df = pd.DataFrame(grid.cv_results_)
output_path = Path("models/hparam_results.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(output_path, index=False)
print(f"Saved hyperparameter CV results to {output_path}")
