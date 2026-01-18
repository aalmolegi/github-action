from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.config import load_params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True, help="Input CSV path")
    parser.add_argument("--out-train", required=True, help="Output train npz path")
    parser.add_argument("--out-test", required=True, help="Output test npz path")
    parser.add_argument("--params", required=True, help="params.yaml path")
    args = parser.parse_args()

    params = load_params(args.params)

    df = pd.read_csv(args.inp)
    if "target" not in df.columns:
        raise ValueError("Expected a 'target' column in the raw dataset.")

    X = df.drop(columns=["target"]).to_numpy(dtype=np.float32)
    y = df["target"].to_numpy(dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params.data.test_size,
        random_state=params.data.random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_test).parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(args.out_train, X=X_train, y=y_train)
    np.savez_compressed(args.out_test, X=X_test, y=y_test)


if __name__ == "__main__":
    main()
