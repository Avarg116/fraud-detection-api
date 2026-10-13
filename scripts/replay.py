import json

import httpx
import joblib

from fraud.training.data import FEATURES
from fraud.training.train import MODEL_DIR, MODEL_VERSION, load_train_val

API_URL = "http://127.0.0.1:8020"

_, val = load_train_val()

# offline
model = joblib.load(MODEL_DIR / f"model-{MODEL_VERSION}.joblib")
threshold = json.loads((MODEL_DIR / f"model-{MODEL_VERSION}.json").read_text())["threshold"]
offline_flags = model.predict_proba(val[FEATURES])[:, 1] >= threshold

# online
mismatches = 0
api_flagged = 0
with httpx.Client(base_url=API_URL) as client:
    for i, (_, row) in enumerate(val.iterrows()):
        tx = {name.lower(): float(row[name]) for name in FEATURES}
        decision = client.post("/predict", json=tx).json()["decision"]
        api_flagged += decision == "FLAG"
        if (decision == "FLAG") != offline_flags[i]:
            mismatches += 1

print("replayed:", len(val), "transactions")
print("API flagged:", api_flagged, "| offline flagged:", offline_flags.sum())
print("mismatches:", mismatches)
