from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.utils.config import load_params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Train npz path")
    parser.add_argument("--model-out", required=True, help="Model output path (joblib)")
    parser.add_argument("--params", required=True, help="params.yaml path")
    args = parser.parse_args()

    params = load_params(args.params)

    data = np.load(args.train)
    X_train = data["X"]
    y_train = data["y"]

    model = LogisticRegression(C=params.model.C, max_iter=params.model.max_iter)
    model.fit(X_train, y_train)

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)


if __name__ == "__main__":
    main()
