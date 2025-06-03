#!/usr/bin/env python3
"""
scripts/backtest_hybrid.py

Backtest the Hybrid ICT+XGB strategy on historical XAUUSD M1 data for the past 6–12 months.
Generates a trade log CSV and prints key metrics: Win Rate, Profit Factor, Max Drawdown, Expectancy.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
import sys

# ─── ปรับ PYTHONPATH ให้รวม project root ─────────────────────────────────────────
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.features import compute_features
from src.ict_signal import generate_ict_signal

# ─── โหลด config ─────────────────────────────────────────────────────────────────
_cfg_path = project_root / "config" / "config.yaml"
with open(_cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

HIST_PATH = Path(cfg["historical_data_path"])
FEAT_PATH = Path(cfg["features_data_path"])
TRADE_LOG_PATH = project_root / "data" / "backtest_trade_log.csv"

# ─── ฟังก์ชันช่วยคำนวณ metrics ────────────────────────────────────────────────────
def compute_metrics(trades_df: pd.DataFrame):
    """
    trades_df ควรมีคอลัมน์: pnl (float), entry_time (datetime), exit_time (datetime)
    คืน dict ของ metrics: win_rate, profit_factor, max_drawdown, expectancy
    """
    # Win Rate
    total_trades = len(trades_df)
    if total_trades == 0:
        return {"win_rate": np.nan, "profit_factor": np.nan,
                "max_drawdown": np.nan, "expectancy": np.nan}

    wins = trades_df[trades_df["pnl"] > 0]["pnl"]
    losses = trades_df[trades_df["pnl"] < 0]["pnl"]

    win_rate = len(wins) / total_trades if total_trades > 0 else np.nan
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf

    # Equity curve & Max Drawdown
    equity = trades_df["pnl"].cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    max_drawdown = drawdown.min()

    # Expectancy: (Avg win * win_rate) - (Avg loss * loss_rate)
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    loss_rate = len(losses) / total_trades
    expectancy = (avg_win * win_rate) + (avg_loss * loss_rate)

    return {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "expectancy": expectancy
    }

# ─── ฟังก์ชันหลักสำหรับ backtest ───────────────────────────────────────────────────
def backtest_hybrid():
    # 1) โหลดข้อมูลย้อนหลัง
    if not HIST_PATH.exists():
        print(f"[{datetime.now()}] Historical data not found at {HIST_PATH}")
        return
    df_hist = pd.read_csv(HIST_PATH, parse_dates=["time"])
    df_hist = df_hist.sort_values("time").reset_index(drop=True)

    # 2) คำนวณฟีเจอร์ใหม่ (เขียนกี่ครั้งก็ได้ เพื่อให้แน่ใจว่าล่าสุด)
    compute_features(str(HIST_PATH), str(FEAT_PATH))

    # 3) โหลด DataFrame ฟีเจอร์
    if not FEAT_PATH.exists():
        print(f"[{datetime.now()}] Features file not found at {FEAT_PATH}")
        return
    df_feat = pd.read_csv(FEAT_PATH, parse_dates=["time"])
    df_feat = df_feat.sort_values("time").reset_index(drop=True)

    # 4) เตรียมเก็บ trade log
    trades = []  # แต่ละรายการเป็น dict: entry_time, exit_time, side, entry_price, exit_price, pnl

    n = len(df_feat)
    idx = 0
    while idx < n:
        # 4.1) ตรวจ ICT signal ที่ idx
        sig = generate_ict_signal(df_feat, idx)
        if sig is None:
            idx += 1
            continue

        # 4.2) เก็บข้อมูล entry
        entry_idx = sig["entry_index"]
        entry_time = sig["entry_time"]
        entry_price = sig["entry_price"]
        side = sig["side"]
        sl = sig["sl"]
        tp1 = sig["tp1"]
        tp2 = sig["tp2"]
        tp3 = sig["tp3"]

        # 4.3) เริ่ม scan bar ถัดไปเพื่อหาจุดออก
        exit_idx = None
        exit_price = None
        pnl = 0.0

        for j in range(entry_idx + 1, n):
            high_j = df_feat.at[j, "high"]
            low_j = df_feat.at[j, "low"]

            # หากเป็น Buy
            if side == "Buy":
                # 4.3.1) TP1 ก่อน
                if high_j >= tp1:
                    exit_idx = j
                    exit_price = tp1
                    pnl = tp1 - entry_price
                    break
                # 4.3.2) TP2
                if high_j >= tp2:
                    exit_idx = j
                    exit_price = tp2
                    pnl = tp2 - entry_price
                    break
                # 4.3.3) TP3
                if high_j >= tp3:
                    exit_idx = j
                    exit_price = tp3
                    pnl = tp3 - entry_price
                    break
                # 4.3.4) SL
                if low_j <= sl:
                    exit_idx = j
                    exit_price = sl
                    pnl = sl - entry_price
                    break
            # หากเป็น Sell
            else:  # side == "Sell"
                # TP1 ก่อน (reverse logic)
                if low_j <= tp1:
                    exit_idx = j
                    exit_price = tp1
                    pnl = entry_price - tp1
                    break
                # TP2
                if low_j <= tp2:
                    exit_idx = j
                    exit_price = tp2
                    pnl = entry_price - tp2
                    break
                # TP3
                if low_j <= tp3:
                    exit_idx = j
                    exit_price = tp3
                    pnl = entry_price - tp3
                    break
                # SL
                if high_j >= sl:
                    exit_idx = j
                    exit_price = sl
                    pnl = entry_price - sl
                    break

        # 4.4) ถ้าไม่เจอ exit ในอนาคต ให้ใช้ราคาปิดสุดท้าย
        if exit_idx is None:
            exit_idx = n - 1
            exit_price = df_feat.at[exit_idx, "close"]
            pnl = (exit_price - entry_price) if side == "Buy" else (entry_price - exit_price)

        exit_time = df_feat.at[exit_idx, "time"]
        trades.append({
            "entry_time": entry_time,
            "exit_time": exit_time,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": round(pnl, 5),
            # เก็บฟีเจอร์ entry สำคัญไว้ด้วย (เช่น ATR, VWAP)
            "atr_entry": df_feat.at[entry_idx, "atr"],
            "vwap_entry": df_feat.at[entry_idx, "vwap"]
        })

        # 4.5) กระโดดไปหลัง exit_idx เพื่อไม่ให้เกิดทับซ้อน
        idx = exit_idx + 1

    # 5) สร้าง DataFrame ของ trade log
    df_trades = pd.DataFrame(trades)
    df_trades.to_csv(TRADE_LOG_PATH, index=False)
    print(f"[{datetime.now()}] Backtest completed. Trade log saved to {TRADE_LOG_PATH}")

    # 6) คำนวณและแสดง metrics
    metrics = compute_metrics(df_trades)
    print("\n===== Backtest Metrics =====")
    print(f"Total Trades   : {len(df_trades)}")
    print(f"Win Rate       : {metrics['win_rate']:.2%}")
    print(f"Profit Factor  : {metrics['profit_factor']:.3f}")
    print(f"Max Drawdown   : {metrics['max_drawdown']:.5f}")
    print(f"Expectancy     : {metrics['expectancy']:.5f}")
    print("============================\n")


if __name__ == "__main__":
    backtest_hybrid()
