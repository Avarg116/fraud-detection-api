import json

import pytest

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": "v1", "threshold": 0.5}

def test_normal_is_approved(client, fake_db, normal_tx):
    r = client.post("/predict", json=normal_tx)
    assert r.status_code == 200
    assert r.json()["decision"] == "APPROVE"
    assert r.json()["fraud_score"] < 0.5
    assert len(fake_db.saved) == 1

def test_fraud_is_flagged(client, fraud_tx):
    r = client.post("/predict", json=fraud_tx)
    assert r.status_code == 200
    assert r.json()["decision"] == "FLAG"

@pytest.mark.parametrize("change", [
    {"amount": -5},
    {"v1": "abc"},
    {"ammount": 10},
    {"v2": None},
])
def test_invalid_input_is_rejected(client, fake_db, normal_tx, change):
    r = client.post("/predict", json={**normal_tx, **change})
    assert r.status_code == 422
    assert fake_db.saved == []

def test_missing_field(client, normal_tx):
    del normal_tx["v14"]
    r = client.post("/predict", json=normal_tx)
    assert r.status_code == 422


def test_nan_is_rejected(client, normal_tx):
    # NaN used to crash the API
    body = json.dumps({**normal_tx, "v3": float("nan")})
    r = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 422
