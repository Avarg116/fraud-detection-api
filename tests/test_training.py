import numpy as np
import pandas as pd
import pytest

from fraud.training.data import time_split


@pytest.fixture
def timeline():
    # 1,000 rows over 1,000 seconds
    times = np.sort(np.random.default_rng(0).integers(0, 1000, size=1000))
    return pd.DataFrame({"Time": times, "Class": 0})

def test_split_no_future_leak(timeline):
    train, val, test = time_split(timeline)
    assert train["Time"].max() < val["Time"].min()
    assert val["Time"].max() < test["Time"].min()

def test_split_keeps_all_rows(timeline):
    train, val, test = time_split(timeline)
    assert len(train) + len(val) + len(test) == len(timeline)
    assert 0.65 < len(train) / len(timeline) < 0.75
