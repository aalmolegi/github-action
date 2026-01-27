from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.datasets import load_breast_cancer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", required=True, help="Output CSV path, e.g. data/raw.csv"
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame  # includes features + target column named 'target'
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
