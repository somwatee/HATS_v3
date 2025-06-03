import xgboost as xgb
import pandas as pd
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# นำ ICT logic เข้ามาใช้
from src.ict_signal import generate_ict_signal

# โหลด config
_cfg_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
with open(_cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

MODEL_PATH = cfg["model_path"]

# ฟีเจอร์คอลัมน์เดียวกับ model_trainer.py
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

class DecisionEngine:
    def __init__(self):
        # โหลดโมเดล XGBoost
        self.clf = xgb.XGBClassifier()
        self.clf.load_model(MODEL_PATH)

    def predict_xgb(self, feature_dict: Dict[str, Any]) -> tuple[str, float]:
        """
        ใช้โมเดล XGBoost ทำนายบน dictionary ของฟีเจอร์ (feature_dict)
        คืน (label, confidence) โดย label ∈ {"Buy","Sell","NoTrade"}
        """
        df = pd.DataFrame([feature_dict])
        X = df[FEATURE_COLS]
        proba = self.clf.predict_proba(X)[0]
        pred_code = int(self.clf.predict(X)[0])
        code_to_label = {0: "NoTrade", 1: "Buy", 2: "Sell"}
        label = code_to_label.get(pred_code, "NoTrade")
        confidence = float(max(proba))
        return label, confidence

    def predict(self, df: pd.DataFrame, idx: int) -> Dict[str, Any]:
        """
        รวม ICT + XGBoost fallback
        1) เรียก generate_ict_signal(df, idx) → ถ้าได้สัญญาณ ICT ให้คืน dict ดังนี้:
           {
             'source': 'ICT',
             'side': 'Buy'/'Sell',
             'entry_index': idx,
             'entry_time': Timestamp,
             'entry_price': float,
             'sl': float,
             'tp1': float,
             'tp2': float,
             'tp3': float,
             'fvg_top': float,
             'fvg_bottom': float,
             'fib_levels': {...},
             'atr': float
           }
        2) ถ้า ICT คืน None → สร้าง feature_dict จาก df.iloc[idx] แล้วเรียก predict_xgb()
           คืน dict ดังนี้:
           {
             'source': 'XGB',
             'side': 'Buy'/'Sell'/'NoTrade',
             'confidence': float
           }
        """
        # 1) ตรวจ ICT entry
        ict_sig = generate_ict_signal(df, idx)
        if ict_sig is not None:
            ict_sig["source"] = "ICT"
            return ict_sig

        # 2) ถ้าไม่มี ICT → เตรียม feature_dict แล้วเรียก XGBoost
        row = df.iloc[idx]
        feature_dict = {col: row[col] for col in FEATURE_COLS}
        label, confidence = self.predict_xgb(feature_dict)
        return {
            "source": "XGB",
            "side": label,
            "confidence": confidence
        }
