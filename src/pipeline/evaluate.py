from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, help="Test npz path")
    parser.add_argument("--model", required=True, help="Model path (joblib)")
    parser.add_argument("--metrics-out", required=True, help="Output metrics JSON path")
    args = parser.parse_args()

    data = np.load(args.test)
    X_test = data["X"]
    y_test = data["y"]

    model = joblib.load(args.model)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }

    out_path = Path(args.metrics_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
