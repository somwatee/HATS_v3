import time
from datetime import datetime
import pandas as pd
import yaml
from pathlib import Path

# ─── ปรับ PYTHONPATH ให้รวม project root ───────────────────────────────────────────────
import sys
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.fetch_candles import fetch_candles
from src.features import compute_features
from src.decision_engine import DecisionEngine
from src.mt5_api import MT5Wrapper
from src.health_report import health_check

# ─── โหลด config ───────────────────────────────────────────────────────────────────────
_cfg_path = project_root / "config" / "config.yaml"
with open(_cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

COOLDOWN   = cfg["cooldown_seconds"]
SYMBOL     = cfg["symbol"]
MT5_CFG    = cfg["mt5"]
HIST_PATH  = cfg["historical_data_path"]
FEAT_PATH  = cfg["features_data_path"]

# ─── เริ่มต้น DecisionEngine และ MT5Wrapper ───────────────────────────────────────────
engine = DecisionEngine()
mt5    = MT5Wrapper(MT5_CFG)

# ─── เก็บตำแหน่งที่เปิดค้างไว้ ────────────────────────────────────────────────────────
open_positions = []

def manage_positions(df_feat: pd.DataFrame):
    """
    ตรวจสอบตำแหน่งที่เปิดค้างไว้:
    - Breakeven หากราคา crossing VWAP
    - ปิดตาม TP1, TP2, TP3
    - SL (market close) ทันทีหากถูก触
    - Reverse MSS: ปิดทันทีหากสัญญาณกลับตัว
    """
    global open_positions
    if not open_positions:
        return

    last = df_feat.iloc[-1]
    price_bid = last["close"]
    vwap      = last["vwap"]
    atr       = last["atr"]

    updated = []
    for pos in open_positions:
        side       = pos["side"]
        entry_price= pos["entry_price"]
        sl         = pos["sl"]
        tp1        = pos["tp1"]
        tp2        = pos["tp2"]
        tp3        = pos["tp3"]
        breakeven  = pos["breakeven"]
        tp1_hit    = pos["tp1_hit"]
        tp2_hit    = pos["tp2_hit"]
        tp3_hit    = pos["tp3_hit"]

        # 1) SL
        if side == "Buy" and price_bid <= sl:
            mt5.close_all(SYMBOL)
            print(f"[{datetime.now()}] SL hit for Buy at {price_bid}. Closed position.")
            continue
        if side == "Sell" and price_bid >= sl:
            mt5.close_all(SYMBOL)
            print(f"[{datetime.now()}] SL hit for Sell at {price_bid}. Closed position.")
            continue

        # 2) Breakeven (VWAP cross)
        if not breakeven:
            if side == "Buy" and price_bid > vwap:
                sl = entry_price
                breakeven = True
                mt5.close_all(SYMBOL)
                print(f"[{datetime.now()}] VWAP crossed for Buy; SL set to breakeven. Closed 50% (simulated).")
            if side == "Sell" and price_bid < vwap:
                sl = entry_price
                breakeven = True
                mt5.close_all(SYMBOL)
                print(f"[{datetime.now()}] VWAP crossed for Sell; SL set to breakeven. Closed 50% (simulated).")

        # 3) TP1
        if not tp1_hit:
            if side == "Buy" and price_bid >= tp1:
                mt5.close_all(SYMBOL)
                tp1_hit = True
                sl = entry_price + 0.5 * atr
                print(f"[{datetime.now()}] TP1 hit for Buy at {price_bid}. Closed 1/3. New SL={sl}.")
            if side == "Sell" and price_bid <= tp1:
                mt5.close_all(SYMBOL)
                tp1_hit = True
                sl = entry_price - 0.5 * atr
                print(f"[{datetime.now()}] TP1 hit for Sell at {price_bid}. Closed 1/3. New SL={sl}.")

        # 4) TP2
        if not tp2_hit:
            if side == "Buy" and price_bid >= tp2:
                mt5.close_all(SYMBOL)
                tp2_hit = True
                print(f"[{datetime.now()}] TP2 hit for Buy at {price_bid}. Closed/all or moved SL to breakeven.")
            if side == "Sell" and price_bid <= tp2:
                mt5.close_all(SYMBOL)
                tp2_hit = True
                print(f"[{datetime.now()}] TP2 hit for Sell at {price_bid}. Closed/all or moved SL to breakeven.")

        # 5) TP3
        if not tp3_hit:
            tp3_level = vwap + 0.5 * atr if side == "Buy" else vwap - 0.5 * atr
            if side == "Buy" and price_bid >= tp3_level:
                mt5.close_all(SYMBOL)
                tp3_hit = True
                print(f"[{datetime.now()}] TP3 hit for Buy at {price_bid}. Closed 1/3 position.")
            if side == "Sell" and price_bid <= tp3_level:
                mt5.close_all(SYMBOL)
                tp3_hit = True
                print(f"[{datetime.now()}] TP3 hit for Sell at {price_bid}. Closed 1/3 position.")

        # 6) Reverse MSS (อ่านค่า hypothetical ใน df_feat)
        bullish_mss = last.get("bullish_mss", False)
        bearish_mss = last.get("bearish_mss", False)
        if side == "Buy" and bearish_mss:
            mt5.close_all(SYMBOL)
            print(f"[{datetime.now()}] Reverse MSS (Bearish) for Buy. Closed position.")
            continue
        if side == "Sell" and bullish_mss:
            mt5.close_all(SYMBOL)
            print(f"[{datetime.now()}] Reverse MSS (Bullish) for Sell. Closed position.")
            continue

        updated.append({
            "side": side,
            "entry_price": entry_price,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "atr": atr,
            "vwap": vwap,
            "breakeven": breakeven,
            "tp1_hit": tp1_hit,
            "tp2_hit": tp2_hit,
            "tp3_hit": tp3_hit
        })

    open_positions[:] = updated


# ─── Loop หลัก ─────────────────────────────────────────────────────────────────────────
while True:
    # 1) Fetch แท่งใหม่ 1 แท่ง
    df_new = fetch_candles(1)
    if df_new is None or df_new.empty:
        time.sleep(COOLDOWN)
        health_check()
        continue

    # 2) Append ลง historical.csv
    hist = pd.read_csv(HIST_PATH, parse_dates=["time"])
    hist = pd.concat([hist, df_new]).drop_duplicates(subset="time") \
               .sort_values("time").reset_index(drop=True)
    hist.to_csv(HIST_PATH, index=False)

    # 3) Recompute features → data_with_features.csv
    compute_features(HIST_PATH, FEAT_PATH)

    # 4) โหลด dataframe ฟีเจอร์ล่าสุด
    df_feat = pd.read_csv(FEAT_PATH, parse_dates=["time"])
    last_idx = len(df_feat) - 1
    last_row = df_feat.iloc[last_idx]

    # 5) สร้างสัญญาณ (ICT หรือ XGB)
    sig = engine.predict(df_feat, last_idx)
    source = sig.get("source")
    side   = sig.get("side")

    print(f"[{datetime.now()}] Signal from {source}: {side}")

    # 6) ถ้า ICT entry เกิด → เปิดออร์เดอร์ + บันทึกตำแหน่ง
    if source == "ICT" and side in ("Buy", "Sell"):
        entry_price = sig["entry_price"]
        sl          = sig["sl"]
        tp1         = sig["tp1"]
        tp2         = sig["tp2"]
        tp3         = sig["tp3"]
        atr         = sig["atr"]
        vwap        = last_row["vwap"]

        # คำนวณ Lot เบื้องต้น (1% equity) → ปรับตาม ATR/Equity จริงได้
        lot = 0.01

        success = mt5.open_order(SYMBOL, side.upper(), lot=lot)
        if success:
            open_positions.append({
                "side": side,
                "entry_price": entry_price,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "tp3": tp3,
                "atr": atr,
                "vwap": vwap,
                "breakeven": False,
                "tp1_hit": False,
                "tp2_hit": False,
                "tp3_hit": False
            })
            print(f"[{datetime.now()}] Opened {side} @ {entry_price}, SL={sl}, TP1={tp1}, TP2={tp2}, TP3={tp3}")

    # 7) จัดการตำแหน่งที่เปิดค้างไว้
    manage_positions(df_feat)

    # 8) ตรวจสุขภาพระบบ ใน loop เดียวกัน
    health_check()

    # 9) พัก COOLDOWN วินาที
    time.sleep(COOLDOWN)
