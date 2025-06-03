import pandas as pd
import numpy as np
import pytest
import xgboost as xgb

from src.decision_engine import DecisionEngine, FEATURE_COLS

def build_dummy_df_for_decision():
    """
    คืน DataFrame ตัวอย่างที่มีคอลัมน์จำเป็นทั้งหมด
    รวมค่าที่ออกแบบให้ idx=25 เกิด ICT signal (Buy)
    และ idx=30 ไม่เกิด ICT เพื่อให้ XGB fallback ทำงาน
    """
    n = 50
    times = pd.date_range("2025-01-01 07:00", periods=n, freq="T")
    df = pd.DataFrame({
        "time": times,
        "open": np.linspace(100, 110, n),
        "high": np.linspace(100.5, 110.5, n),
        "low": np.linspace(99.5, 109.5, n),
        "close": np.linspace(100.2, 110.2, n),
        "tick_volume": np.random.randint(50, 200, size=n),
        "atr": [0.2] * n,
        "vwap": np.linspace(100.2, 110.2, n),
        "ema50_h4": [105] * n,
        "ema200_h4": [100] * n,
        "rsi_h4": [55] * n,
    })

    # สร้าง swing_low ที่ idx=20, swing_high ที่ idx=22
    df.at[20, "low"] = 95
    df.at[22, "high"] = 115
    # MSS bullish ที่ idx=23 (close > last swing_high=115)
    df.at[23, "close"] = 116
    # FVG bullish ที่ idx=24: 3 แท่งเขียวติดต่อกัน
    df.at[21, "open"], df.at[21, "close"] = 101, 102
    df.at[22, "open"], df.at[22, "close"] = 103, 104
    df.at[23, "open"], df.at[23, "close"] = 105, 106
    # กำหนดให้เกิด gap สำหรับ FVG
    df.at[22, "low"] = 110
    df.at[21, "high"] = 108
    # ให้ bar idx=24 อยู่ในโซน pullback
    df.at[24, "open"], df.at[24, "close"] = 109, 109.5
    # ให้ idx=25 มีราคาเปิด-ปิด เพื่อสร้าง entry
    df.at[25, "open"], df.at[25, "close"] = 109.5, 110

    # เพิ่มคอลัมน์ ICT ที่ generate_ict_signal คาดหวัง แต่จะไม่ถูกใช้เพราะเรา monkeypatch
    df["last_swing_low"] = np.nan
    df["last_swing_high"] = np.nan
    df["bullish_fvg"] = False
    df["bearish_fvg"] = False
    df["fvg_top"] = np.nan
    df["fvg_bottom"] = np.nan

    # เพิ่มคอลัมน์ feature อื่นๆ (ที่ FEATURE_COLS ระบุ) ด้วยค่าเริ่มต้น 0
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    return df

@pytest.fixture(autouse=True)
def patch_load_model(monkeypatch, tmp_path):
    """
    สร้าง XGBClassifier dummy ให้ DecisionEngine โหลดแทนโมเดลจริง
    """
    dummy_clf = xgb.XGBClassifier(objective="multi:softprob", num_class=3, random_state=42)
    # สร้าง DataFrame dummy ตาม FEATURE_COLS เพื่อ fit
    X_dummy = pd.DataFrame([{col: 0 for col in FEATURE_COLS}])
    y_dummy = pd.Series([0])
    dummy_clf.fit(X_dummy, y_dummy)

    model_path = tmp_path / "dummy_model.json"
    dummy_clf.save_model(str(model_path))

    # Monkeypatch MODEL_PATH ในโมดูล decision_engine ให้ชี้ไป dummy_model.json
    import src.decision_engine as de_mod
    monkeypatch.setattr(de_mod, "MODEL_PATH", str(model_path))

    yield

def test_decision_engine_ict_priority(monkeypatch):
    """
    เมื่อ idx ตรงเงื่อนไข ICT (idx=25) → predict() ควรคืน source="ICT" และ side="Buy"
    """
    df = build_dummy_df_for_decision()

    # monkeypatch generate_ict_signal ใน namespace ของ decision_engine
    import src.decision_engine as de_mod
    def fake_generate_ict(df_local, idx):
        if idx == 25:
            return {
                "side": "Buy",
                "entry_index": 25,
                "entry_time": df_local.at[25, "time"],
                "entry_price": df_local.at[25, "open"],
                "sl": df_local.at[25, "open"] - 0.5 * df_local.at[25, "atr"],
                "tp1": 115 + 1.272 * (115 - 95),
                "tp2": df_local.at[25, "open"] + 2 * df_local.at[25, "atr"],
                "tp3": df_local.at[25, "vwap"] + 0.5 * df_local.at[25, "atr"],
                "fvg_top": 110,
                "fvg_bottom": 108,
                "fib_levels": {
                    "fib_382": 115 - 0.382 * 20,
                    "fib_50": 115 - 0.5 * 20,
                    "fib_618": 115 - 0.618 * 20,
                    "ext_1272": 95 + 1.272 * 20
                },
                "atr": df_local.at[25, "atr"]
            }
        return None

    monkeypatch.setattr(de_mod, "generate_ict_signal", fake_generate_ict)

    engine = DecisionEngine()
    result = engine.predict(df, idx=25)
    assert result["source"] == "ICT"
    assert result["side"] == "Buy"
    assert "entry_price" in result and "sl" in result

def test_decision_engine_xgb_fallback(monkeypatch):
    """
    เมื่อไม่มี ICT (idx != 25) → predict() ควร fallback ไป XGBoost
    """
    df = build_dummy_df_for_decision()

    import src.decision_engine as de_mod
    monkeypatch.setattr(de_mod, "generate_ict_signal", lambda df_local, idx: None)

    dummy_pred = np.array([2])
    dummy_proba = np.array([[0.1, 0.2, 0.7]])
    monkeypatch.setattr(xgb.XGBClassifier, "predict", lambda self, X: dummy_pred)
    monkeypatch.setattr(xgb.XGBClassifier, "predict_proba", lambda self, X: dummy_proba)

    engine = DecisionEngine()
    result = engine.predict(df, idx=30)
    assert result["source"] == "XGB"
    assert result["side"] == "Sell"
    assert abs(result["confidence"] - 0.7) < 1e-6
