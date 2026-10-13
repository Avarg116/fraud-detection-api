import json
import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn
from fastapi.testclient import TestClient

import fraud.api.app as app_module
from fraud.training.data import FEATURES
from fraud.training.train import build_model

TEST_THRESHOLD = 0.5

@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    # small model trained on random data. Very negative V14 = fraud
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(2000, len(FEATURES))), columns=FEATURES)

    y = (df["V14"] < -1.65).astype(int)

    model = build_model().fit(df[FEATURES], y)

    path = tmp_path_factory.mktemp("models")
    joblib.dump(model, path / "model-v1.joblib")

    metadata = {
        "model_version": "v1",
        "sklearn_version": sklearn.__version__,
        "features": FEATURES,
        "threshold": TEST_THRESHOLD,
    }
    (path / "model-v1.json").write_text(json.dumps(metadata))
    return path


@pytest.fixture
def normal_tx(): 
    tx = {f"v{i}": 0.0 for i in range(1, 29)} # normal transacstion
    tx["amount"] = 25.0
    return tx

@pytest.fixture
def fraud_tx(normal_tx):
    return {**normal_tx, "v14": -6.0} # fraud transaction

class FakeDB:
    def __init__(self):
        self.saved = []

    def save_prediction(self, pool, tx, fraud_score, decision, model_version, threshold):
        self.saved.append(decision)
        return len(self.saved)

class FakePool:
    def close(self):
        pass

@pytest.fixture
def fake_db():
    return FakeDB()


@pytest.fixture
def client(monkeypatch, model_dir, fake_db):
    # using the test model and fake db
    monkeypatch.setattr(app_module, "MODEL_DIR", model_dir)
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    monkeypatch.setattr(app_module, "open_pool", lambda url: FakePool())
    monkeypatch.setattr(app_module, "save_prediction", fake_db.save_prediction)

    with TestClient(app_module.app) as c:
        yield c
