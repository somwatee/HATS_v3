import pandas as pd
import yaml
from pathlib import Path

# โหลด ICT logic (ต้องมีไฟล์ src/ict_signal.py พร้อมใช้งาน)
from src.ict_signal import detect_swing_points, detect_mss, compute_fvg, generate_ict_signal

# โหลด config
_cfg_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
with open(_cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# พาธไฟล์ที่ต้องใช้ / สร้าง
features_path = Path(cfg["features_data_path"])      # data/data_with_features.csv
output_path   = Path(cfg["historical_data_path"]).parent / "with_labels_ict.csv"  # data/with_labels_ict.csv

def label_ict(input_path: str, output_path: str):
    """
    อ่านไฟล์ features (data_with_features.csv) → ใช้ ICT Logic สร้าง label “Buy”/“Sell”/“NoTrade”
    แล้วบันทึกเป็น data/with_labels_ict.csv
    """
    # 1) โหลด DataFrame ฟีเจอร์ (M1) ทั้งหมด
    df = pd.read_csv(input_path, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # 2) เตรียม DataFrame: คำนวณ swing points, MSS, FVGล่วงหน้า
    #    (ฟังก์ชันเหล่านี้ return df ที่มีคอลัมน์เสริมสำหรับ ICT)
    df = detect_swing_points(df, window=5)
    df = detect_mss(df)
    df = compute_fvg(df)

    # 3) สร้างลิสต์เก็บ label เริ่มต้นทุกแถวเป็น "NoTrade"
    labels = ["NoTrade"] * len(df)

    # 4) วนลูปทุก index → เรียก generate_ict_signal
    for i in range(len(df)):
        sig = generate_ict_signal(df, idx=i)
        if sig is not None:
            # กำหนด label ตาม 'side' (Buy/Sell)
            labels[i] = sig["side"]

    # 5) แปะคอลัมน์ label ลงใน df
    df["label"] = labels

    # 6) บันทึกเป็น CSV ใหม่ (รวมทั้งคอลัมน์ features เดิม + label)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Labels (ICT) saved to {output_path}")

if __name__ == "__main__":
    label_ict(str(features_path), str(output_path))
