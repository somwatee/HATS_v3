import pandas as pd
import yaml
from pathlib import Path

# โหลด config จาก config/config.yaml
_cfg_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
with open(_cfg_path, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

# ค่า Horizon และ ATR multiplier สำหรับการตั้ง threshold
H = _cfg.get("label_horizon", 5)
k_atr = _cfg.get("label_atr_multiplier", 0.5)

def build_labels(input_path: str, output_path: str = "data/with_labels.csv"):
    """
    อ่านไฟล์ features (data_with_features.csv) → สร้าง label “Buy”/“Sell”/“NoTrade”
    ตามเงื่อนไข
    1) Base label จาก ATR-break ภายใน H แท่งถัดไป
    2) กรองด้วย Higher-Timeframe & VWAP bias
    3) กรองด้วย Bollinger Bands + ATR_MA
    4) กรองด้วย MSS/FVG + Fibonacci + RSI/ADX + Volume Imbalance
    จากนั้นบันทึกลง output_path
    """
    # 1. โหลด DataFrame ฟีเจอร์
    df = pd.read_csv(input_path, parse_dates=["time"])
    n = len(df)
    labels = ["NoTrade"] * n

    for t in range(n):
        # ถ้าไม่พอช่วงถัดไป ให้เป็น NoTrade
        if t + H >= n:
            labels[t] = "NoTrade"
            continue

        # 1) Base label จาก ATR-break
        price_open = df.at[t, "open"]
        atr_t = df.at[t, "atr"]
        threshold = k_atr * atr_t

        future_high = df["high"].iloc[t + 1 : t + 1 + H].max()
        future_low  = df["low"].iloc[t + 1 : t + 1 + H].min()

        if (future_high - price_open) >= threshold:
            base_label = "Buy"
        elif (price_open - future_low) >= threshold:
            base_label = "Sell"
        else:
            labels[t] = "NoTrade"
            continue

        # 2) HTF + VWAP bias
        ema50_h4 = df.at[t, "ema50_h4"]
        ema200_h4 = df.at[t, "ema200_h4"]
        rsi_h4 = df.at[t, "rsi_h4"]
        vwap = df.at[t, "vwap"]
        tol = 0.1 * atr_t

        if base_label == "Buy":
            cond_htf = (ema50_h4 > ema200_h4) and (rsi_h4 > 50)
            cond_vwap = (price_open > vwap + tol)
            if not (cond_htf and cond_vwap):
                labels[t] = "NoTrade"
                continue
        else:  # Sell
            cond_htf = (ema50_h4 < ema200_h4) and (rsi_h4 < 50)
            cond_vwap = (price_open < vwap - tol)
            if not (cond_htf and cond_vwap):
                labels[t] = "NoTrade"
                continue

        # 3) Bollinger Bands + ATR_MA filter
        bb_upper = df.at[t, "bb_upper"]
        bb_lower = df.at[t, "bb_lower"]
        atr_ma = df.at[t, "atr_ma"]

        if base_label == "Buy":
            # ถ้า price_open <= bb_lower และ ATR < ATR_MA หรือ price_open >= bb_upper และ ATR > ATR_MA → ผ่าน
            if price_open <= bb_lower and atr_t < atr_ma:
                pass
            elif price_open >= bb_upper and atr_t > atr_ma:
                pass
            else:
                labels[t] = "NoTrade"
                continue
        else:  # Sell
            if price_open >= bb_upper and atr_t < atr_ma:
                pass
            elif price_open <= bb_lower and atr_t > atr_ma:
                pass
            else:
                labels[t] = "NoTrade"
                continue

        # 4) MSS/FVG + Fibonacci + RSI/ADX + Volume Imbalance
        mss_b = df.at[t, "mss_bullish"]
        mss_br = df.at[t, "mss_bearish"]
        fvg_b = df.at[t, "fvg_bullish"]
        fvg_br = df.at[t, "fvg_bearish"]
        fib_in_zone = df.at[t, "fib_in_zone"]
        rsi_t = df.at[t, "rsi"]
        adx_t = df.at[t, "adx"]
        vol_imb = df.at[t, "vol_imbalance"]

        if base_label == "Buy":
            cond1 = mss_b or (fvg_b and fib_in_zone)
            cond2 = (rsi_t < 30 and adx_t > 25) or (vol_imb > 0.2)
            if cond1 and cond2:
                labels[t] = "Buy"
            else:
                labels[t] = "NoTrade"
        else:  # Sell
            cond1 = mss_br or (fvg_br and fib_in_zone)
            cond2 = (rsi_t > 70 and adx_t > 25) or (vol_imb < -0.2)
            if cond1 and cond2:
                labels[t] = "Sell"
            else:
                labels[t] = "NoTrade"

    # แปลง labels เป็นคอลัมน์ใหม่ใน DataFrame
    df["label"] = labels

    # สร้างโฟลเดอร์ปลายทาง (ถ้ายังไม่มี) แล้วบันทึก CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Labels saved to {output_path}")

if __name__ == "__main__":
    build_labels("data/data_with_features.csv", "data/with_labels.csv")
