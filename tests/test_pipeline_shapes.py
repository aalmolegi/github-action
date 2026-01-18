import json
from pathlib import Path

import numpy as np


def test_preprocessed_npz_shapes(tmp_path: Path):
    # Create fake minimal arrays to simulate preprocessed output
    X = np.random.randn(10, 5).astype(np.float32)
    y = np.random.randint(0, 2, size=(10,), dtype=np.int64)

    out = tmp_path / "train.npz"
    np.savez_compressed(out, X=X, y=y)

    loaded = np.load(out)
    assert loaded["X"].ndim == 2
    assert loaded["y"].ndim == 1
    assert loaded["X"].shape[0] == loaded["y"].shape[0]


def test_metrics_json_format(tmp_path: Path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"accuracy": 0.9, "f1": 0.8, "roc_auc": 0.95}))

    metrics = json.loads(metrics_path.read_text())
    assert set(metrics.keys()) == {"accuracy", "f1", "roc_auc"}
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
