# tests/test_run_phase3.py

import os
import time
import pytest
import pandas as pd
import numpy as np
import runpy
from pathlib import Path

class BreakLoop(Exception):
    pass

def test_run_phase3_single_iteration(tmp_path, monkeypatch):
    """
    ทดสอบว่า run_phase3.py จะ:
    1) เรียก fetch_candles, compute_features
    2) เรียก DecisionEngine.predict แล้วเปิดออร์เดอร์ผ่าน MT5Wrapper.open_order
    3) เรียก manage_positions, health_check แล้วหยุด loop หลัง iteration แรก
    """
    # 1) สร้างโครงสร้างโปรเจกต์จำลองใน tmp_path
    project_root = tmp_path
    (project_root / "config").mkdir()
    (project_root / "scripts").mkdir()
    (project_root / "src").mkdir()

    # คัดลอกสคริปต์ run_phase3.py จาก workspace จริงมาไว้ใน tmp_path/scripts
    import shutil
    current_root = Path(__file__).resolve().parents[2]
    shutil.copy(current_root / "scripts" / "run_phase3.py", project_root / "scripts" / "run_phase3.py")

    # 2) สร้าง config/config.yaml
    cfg = {
        "cooldown_seconds": 0,
        "symbol": "TESTSYM",
        "mt5": {
            "terminal_path": "dummy_path",
            "login": 0,
            "server": "dummy",
            "password": "dummy",
            "timeout": 1000
        },
        "historical_data_path": str(project_root / "data" / "historical.csv"),
        "features_data_path": str(project_root / "data" / "data_with_features.csv")
    }
    import yaml
    with open(project_root / "config" / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    # 3) สร้างโฟลเดอร์ data และไฟล์ historical.csv + data_with_features.csv
    (project_root / "data").mkdir()
    df_hist = pd.DataFrame(columns=["time","open","high","low","close","tick_volume"])
    df_hist.to_csv(project_root / "data" / "historical.csv", index=False)

    df_feat = pd.DataFrame([{
        "time": pd.Timestamp("2025-01-01 00:00"),
        "open": 100, "high": 101, "low": 99, "close": 100.5, "tick_volume": 100,
        "atr": 0.5, "vwap": 100.5,
        "ema50_h4": 100, "ema200_h4": 95, "rsi_h4": 60,
        "bb_upper": 102, "bb_lower": 98, "atr_ma": 0.5,
        "bb_upper_diff": 0, "bb_lower_diff": 0, "vol_imbalance": 0,
        "mss_bullish": False, "mss_bearish": False, "fvg_bullish": False, "fvg_bearish": False
    }])
    (project_root / "data" / "data_with_features.csv").parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(project_root / "data" / "data_with_features.csv", index=False)

    # 4) เปลี่ยน working directory ไปยัง project_root
    monkeypatch.chdir(project_root)

    # 5) Monkeypatch ฟังก์ชันต่าง ๆ ใน src
    # - fetch_candles ให้คืน DataFrame ใหม่เสมอ
    def fake_fetch(n):
        return pd.DataFrame([{
            "time": pd.Timestamp("2025-01-01 00:01"),
            "open": 100, "high": 101, "low": 99, "close": 100.5, "tick_volume": 100
        }])
    monkeypatch.setattr("src.fetch_candles.fetch_candles", fake_fetch)

    # - compute_features ให้ no-op (ใช้ไฟล์ features ที่เตรียมไว้แล้ว)
    def fake_compute(in_path, out_path):
        pass
    monkeypatch.setattr("src.features.compute_features", fake_compute)

    # - DecisionEngine.predict ให้คืน ICT signal ทุกครั้ง
    fake_signal = {
        "source": "ICT",
        "side": "Buy",
        "entry_index": 1,
        "entry_time": pd.Timestamp("2025-01-01 00:01"),
        "entry_price": 100.5,
        "sl": 100.0,
        "tp1": 102.0,
        "tp2": 103.0,
        "tp3": 101.0,
        "atr": 0.5
    }
    monkeypatch.setattr("src.decision_engine.DecisionEngine.predict", lambda self, df, idx: fake_signal)

    # - MT5Wrapper.open_order ให้ return True และบันทึกว่าเรียกถูกต้อง
    calls = {"open_order": False}
    def fake_open(symbol, side, lot=0.01, sl=None, tp=None):
        calls["open_order"] = True
        return True
    monkeypatch.setattr("src.mt5_api.MT5Wrapper.open_order", fake_open)

    # - MT5Wrapper.close_all ให้ no-op
    monkeypatch.setattr("src.mt5_api.MT5Wrapper.close_all", lambda self, sym: True)

    # - health_check ให้ no-op
    monkeypatch.setattr("src.health_report.health_check", lambda: None)

    # - time.sleep ให้โยน BreakLoop เพื่อหยุด loop หลัง iteration แรก
    monkeypatch.setattr(time, "sleep", lambda s: (_ for _ in ()).throw(BreakLoop()))

    # 6) เรียก run_phase3.py ผ่าน runpy.run_path() แล้วจับ BreakLoop
    with pytest.raises(BreakLoop):
        runpy.run_path(str(project_root / "scripts" / "run_phase3.py"), run_name="__main__")

    # 7) ตรวจว่ามีการเรียก MT5Wrapper.open_order อย่างน้อยหนึ่งครั้ง
    assert calls["open_order"], "Expected MT5Wrapper.open_order to be called"
