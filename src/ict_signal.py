import pandas as pd
import numpy as np
import talib
from datetime import time
from typing import Optional, Dict

# ─── พารามิเตอร์หลัก (สามารถปรับได้ใน config ในอนาคต) ─── #
ALPHA_ATR = 0.5      # α สำหรับ SL offset
BETA_ATR  = 1.0      # β สำหรับ ATR Trailing factor
SESSION_START = time(7, 0)   # 07:00 GMT+7
SESSION_END   = time(15, 0)  # 15:00 GMT+7
# ──────────────────────────────────────────────────────────── #

def is_in_session(ts: pd.Timestamp) -> bool:
    """
    คืน True ก็ต่อเมื่อ timestamp อยู่ในช่วง SESSION_START–SESSION_END (GMT+7)
    """
    t = ts.to_pydatetime().time()
    return SESSION_START <= t <= SESSION_END

def detect_swing_points(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    ตรวจหา Swing High / Swing Low บน DF ที่มีคอลัมน์ ['time','open','high','low','close','tick_volume']
    - Swing High: high ของแท่งนั้นเป็นค่าสูงสุดในช่วง ±(window//2)
    - Swing Low: low ของแท่งนั้นเป็นค่าต่ำสุดในช่วง ±(window//2)

    คืน df ที่มีคอลัมน์ ['is_swing_high','is_swing_low']
    """
    n = len(df)
    df = df.copy().reset_index(drop=True)
    df["is_swing_high"] = False
    df["is_swing_low"] = False
    half = window // 2

    for i in range(half, n - half):
        high_window = df["high"].iloc[i - half : i + half + 1]
        low_window  = df["low"].iloc[i - half : i + half + 1]
        if df["high"].iloc[i] == high_window.max():
            df.at[i, "is_swing_high"] = True
        if df["low"].iloc[i] == low_window.min():
            df.at[i, "is_swing_low"] = True

    return df

def detect_mss(df: pd.DataFrame) -> pd.DataFrame:
    """
    ตรวจ Market Structure Shift (MSS) บน M1/M5
    - Bullish MSS: Close ปัจจุบันทะลุ Last Swing High
    - Bearish MSS: Close ปัจจุบันทะลุ Last Swing Low

    ต้องเรียกหลัง detect_swing_points() แล้วมีคอลัมน์ is_swing_high, is_swing_low
    คืน df ที่เพิ่มคอลัมน์ ['bullish_mss','bearish_mss','last_swing_high','last_swing_low']
    """
    df = df.copy().reset_index(drop=True)
    last_sh = None
    last_sl = None

    bullish_mss = []
    bearish_mss = []
    last_sh_list = []
    last_sl_list = []

    for i in range(len(df)):
        if df.at[i, "is_swing_high"]:
            last_sh = df.at[i, "high"]
        if df.at[i, "is_swing_low"]:
            last_sl = df.at[i, "low"]

        if last_sh is not None and df.at[i, "close"] > last_sh:
            bullish_mss.append(True)
        else:
            bullish_mss.append(False)

        if last_sl is not None and df.at[i, "close"] < last_sl:
            bearish_mss.append(True)
        else:
            bearish_mss.append(False)

        last_sh_list.append(last_sh if last_sh is not None else np.nan)
        last_sl_list.append(last_sl if last_sl is not None else np.nan)

    df["bullish_mss"] = bullish_mss
    df["bearish_mss"] = bearish_mss
    df["last_swing_high"] = last_sh_list
    df["last_swing_low"]  = last_sl_list
    return df

def compute_fvg(df: pd.DataFrame) -> pd.DataFrame:
    """
    ตรวจ Fair Value Gap (FVG) ของแต่ละแท่ง (simplified)
    กฎ:
      - Bullish FVG: 3 แท่งเขียวต่อเนื่อง (close > open)
        แล้วช่องว่าง (gap) ระหว่าง low(prev1) กับ high(prev2)
      - Bearish FVG: 3 แท่งแดงต่อเนื่อง (close < open)
        แล้วช่องว่างระหว่าง high(prev1) กับ low(prev2)

    คืน df ที่เพิ่มคอลัมน์ ['bullish_fvg','bearish_fvg','fvg_top','fvg_bottom']
    """
    df = df.copy().reset_index(drop=True)
    n = len(df)
    df["bullish_fvg"] = False
    df["bearish_fvg"] = False
    df["fvg_top"] = np.nan
    df["fvg_bottom"] = np.nan

    for i in range(3, n):
        c1, c2, c3 = df.at[i-3, "close"], df.at[i-2, "close"], df.at[i-1, "close"]
        o1, o2, o3 = df.at[i-3, "open"],  df.at[i-2, "open"],  df.at[i-1, "open"]

        # 3 แท่งเขียวต่อเนื่อง
        if (c1 > o1) and (c2 > o2) and (c3 > o3):
            low_prev1  = df.at[i-2, "low"]
            high_prev2 = df.at[i-3, "high"]
            if low_prev1 > high_prev2:
                df.at[i, "bullish_fvg"] = True
                df.at[i, "fvg_bottom"] = high_prev2
                df.at[i, "fvg_top"] = low_prev1

        # 3 แท่งแดงต่อเนื่อง
        if (c1 < o1) and (c2 < o2) and (c3 < o3):
            high_prev1 = df.at[i-2, "high"]
            low_prev2  = df.at[i-3, "low"]
            if high_prev1 < low_prev2:
                df.at[i, "bearish_fvg"] = True
                df.at[i, "fvg_top"] = low_prev2
                df.at[i, "fvg_bottom"] = high_prev1

    return df

def compute_fibonacci_levels(swing_low: float, swing_high: float) -> Dict[str, float]:
    """
    คืนค่า Fibonacci Retracement levels (38.2%, 50%, 61.8%) และ Extension (127.2%)
    keys: ["fib_382","fib_50","fib_618","ext_1272"]
    """
    diff = swing_high - swing_low
    return {
        "fib_382": swing_high - 0.382 * diff,
        "fib_50":  swing_high - 0.5   * diff,
        "fib_618": swing_high - 0.618 * diff,
        "ext_1272": swing_low + 1.272 * diff
    }

def generate_ict_signal(df: pd.DataFrame, idx: int) -> Optional[Dict]:
    """
    ตรวจแท่งที่ idx ว่าตรงเงื่อนไข ICT entry หรือไม่
    เงื่อนไขหลัก:
      1) Session filter (07:00–15:00)
      2) HTF filter: EMA50_H4 vs EMA200_H4 และ RSI_H4
      3) เคยเกิด MSS (last_swing_low/high ไม่ใช่ NaN)
      4) มี FVG ณ idx (bullish_fvg หรือ bearish_fvg)
      5) FVG อยู่ในช่วงโซน Fibonacci (61.8–50 หรือ 50–38.2)
      6) Pullback: bar แถว idx (open หรือ close) อยู่ในโซน FVG ± (0.5×ATR)
      7) คืน dict {'side','entry_index','entry_time','entry_price','sl','tp1','tp2','tp3','fvg_top','fvg_bottom','fib_levels','atr'}
      8) ถ้าไม่เข้าเงื่อนไขใด คืน None

    df ต้องมีคอลัมน์:
    ['time','open','high','low','close','tick_volume','atr','vwap',
      'ema50_h4','ema200_h4','rsi_h4','bullish_mss','bearish_mss',
      'last_swing_low','last_swing_high','bullish_fvg','bearish_fvg',
      'fvg_top','fvg_bottom']
    """
    row = df.iloc[idx]
    ts   = row["time"]

    # 1) Session filter
    if not is_in_session(ts):
        return None

    # 2) HTF filter
    ema50_h4 = row["ema50_h4"]
    ema200_h4 = row["ema200_h4"]
    rsi_h4 = row["rsi_h4"]
    htf_buy = (ema50_h4 > ema200_h4) and (rsi_h4 > 50)
    htf_sell = (ema50_h4 < ema200_h4) and (rsi_h4 < 50)
    if not (htf_buy or htf_sell):
        return None

    # 3) MSS ต้องเคยเกิดก่อนหน้า
    swing_low  = row["last_swing_low"]
    swing_high = row["last_swing_high"]
    if pd.isna(swing_low) or pd.isna(swing_high):
        return None

    # 4) ตรวจ FVG ณ idx
    bullish_fvg = row["bullish_fvg"]
    bearish_fvg = row["bearish_fvg"]
    fvg_top    = row["fvg_top"]
    fvg_bottom = row["fvg_bottom"]
    if not (bullish_fvg or bearish_fvg):
        return None

    # 5) คำนวณ Fibonacci รอบ swing_low ↔ swing_high
    fibs = compute_fibonacci_levels(swing_low, swing_high)

    # ตรวจว่า FVG top/bottom อยู่ในโซน fib
    in_fibo_zone = False
    if bullish_fvg:
        top = fvg_top
        if (fibs["fib_618"] >= top >= fibs["fib_50"]) or (fibs["fib_50"] >= top >= fibs["fib_382"]):
            in_fibo_zone = True
    if bearish_fvg:
        bot = fvg_bottom
        if (fibs["fib_618"] <= bot <= fibs["fib_50"]) or (fibs["fib_50"] <= bot <= fibs["fib_382"]):
            in_fibo_zone = True
    if not in_fibo_zone:
        return None

    # 6) ตรวจ Pullback: bar เปิดหรือปิด อยู่ในโซน FVG ± (0.5×ATR)
    atr = row["atr"]
    buffer = 0.5 * atr
    price_open  = row["open"]
    price_close = row["close"]

    if bullish_fvg:
        zone_low  = fvg_bottom - buffer
        zone_high = fvg_top + buffer
        if not (zone_low <= price_open <= zone_high or zone_low <= price_close <= zone_high):
            return None
        side = "Buy"
        entry_price = price_open
        sl = fvg_bottom - ALPHA_ATR * atr
        tp1 = fibs["ext_1272"]
        tp2 = entry_price + 2 * atr
        vwap = row["vwap"]
        tp3 = vwap + 0.5 * atr

    else:  # bearish_fvg
        zone_low  = fvg_bottom - buffer
        zone_high = fvg_top + buffer
        if not (zone_low <= price_open <= zone_high or zone_low <= price_close <= zone_high):
            return None
        side = "Sell"
        entry_price = price_open
        sl = fvg_top + ALPHA_ATR * atr
        tp1 = fibs["ext_1272"]
        tp2 = entry_price - 2 * atr
        vwap = row["vwap"]
        tp3 = vwap - 0.5 * atr

    return {
        "side": side,
        "entry_index": idx,
        "entry_time": ts,
        "entry_price": entry_price,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "fvg_top": fvg_top,
        "fvg_bottom": fvg_bottom,
        "fib_levels": fibs,
        "atr": atr
    }
