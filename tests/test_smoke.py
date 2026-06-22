"""Day-one smoke tests (CPU, tiny SAE, a few steps)."""
import pytest

torch = pytest.importorskip("torch")

from src import train, evaluate, run_experiments


def test_train_smoke():
    loss = train.smoke()
    assert loss == loss  # not NaN


def test_eval_and_run_smoke():
    out = evaluate.smoke()
    assert 0.0 <= out["cfs"] <= 1.0
    rows = run_experiments.smoke()
    assert rows and rows[0]["variant"] == "naive_steer"
    for r in rows:                       # CFS is the headline number per variant
        assert 0.0 <= r["cfs"] <= 1.0
        assert 0.0 <= r["cfs_empirical"] <= 1.0
