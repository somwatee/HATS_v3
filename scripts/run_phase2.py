import yaml
from pathlib import Path

# ปรับ PYTHONPATH ให้รวม src/
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model_trainer import train_walkforward
# (ถ้าต้องการรัน tune_model ด้วย ก็ import ได้: from src.tune_model import ...)

def main():
    # โหลด config
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_path = cfg["dataset_path"]            # ปกติคือ "data/with_labels_ict.csv"
    model_output  = cfg["model_path"]              # เช่น "models/xgb_hybrid_trading.json"
    report_output = str(Path(cfg["model_path"]).parent / "walkforward_report.txt")

    # (ถ้าต้องการรัน GridSearchCV ก่อน ให้ uncomment บรรทัดนี้)
    # from src.tune_model import grid
    # print(">>> Running hyperparameter tuning (GridSearchCV)...")
    # grid()

    print(">>> Phase 2: Training XGBoost with Walk‐forward CV")
    train_walkforward(dataset_path, model_output, report_output)

if __name__ == "__main__":
    main()
