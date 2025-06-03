# ไฟล์: scripts/run_phase1.py

import sys
from pathlib import Path

# ─── เพิ่ม project root (folder บนสุด) ใน sys.path ───
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.fetch_candles import fetch_candles
from src.features import compute_features
from src.label_ict import label_ict
import yaml

def main():
    # โหลด config
    _cfg_path = project_root / "config" / "config.yaml"
    with open(_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Phase 1.1: Fetch candles
    print(">>> Phase 1.1: Fetching candles")
    hist_path = cfg["historical_data_path"]
    df_new = fetch_candles(cfg["fetch_candles_n"])
    if df_new is not None and not df_new.empty:
        hist_file = Path(hist_path)
        hist_file.parent.mkdir(parents=True, exist_ok=True)
        if hist_file.exists():
            import pandas as pd
            df_old = pd.read_csv(hist_path, parse_dates=["time"])
            df_concat = pd.concat([df_old, df_new]).drop_duplicates(subset="time").sort_values("time")
            df_concat.to_csv(hist_path, index=False)
            print(f"Appended {len(df_new)} rows → {hist_path}")
        else:
            df_new.to_csv(hist_path, index=False)
            print(f"Saved {len(df_new)} rows → {hist_path}")
    else:
        print("No new candles fetched or fetch failed.")

    # Phase 1.2: Compute features
    print("\n>>> Phase 1.2: Computing features")
    features_in  = cfg["historical_data_path"]
    features_out = cfg["features_data_path"]
    compute_features(str(features_in), str(features_out))

    # Phase 1.3: Generate ICT-based labels
    print("\n>>> Phase 1.3: Generating ICT labels")
    labels_in  = cfg["features_data_path"]
    labels_out = str(Path(cfg["features_data_path"]).parent / "with_labels_ict.csv")
    label_ict(str(labels_in), labels_out)

if __name__ == "__main__":
    main()
