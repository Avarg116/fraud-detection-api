"""Pick the decision threshold on the validation set."""

import argparse
import json
import joblib
import numpy as np

from fraud.training.data import FEATURES, TARGET
from fraud.training.train import MODEL_DIR, MODEL_VERSION, load_train_val

CANDIDATES = [0.5, 0.7, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995]


def confusion_at(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    flagged = scores >= threshold
    tp = int((flagged & (y == 1)).sum())  # frauds caught
    fp = int((flagged & (y == 0)).sum())  # false alarms
    fn = int((~flagged & (y == 1)).sum())  # frauds missed
    return {
        "threshold": threshold,
        "flagged": tp + fp,
        "caught": tp,
        "missed": fn,
        "false_alarms": fp,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", type=float, help="save this threshold into the model metadata")
    args = parser.parse_args()

    _train, val = load_train_val()
    model = joblib.load(MODEL_DIR / f"model-{MODEL_VERSION}.joblib")
    y = val[TARGET].to_numpy()
    scores = model.predict_proba(val[FEATURES])[:, 1]

    print("validation:", len(y), "transactions,", y.sum(), "frauds")
    for t in CANDIDATES:
        r = confusion_at(y, scores, t)
        print(t, "flagged:", r["flagged"], "caught:", r["caught"], "false alarms:", r["false_alarms"],
              "precision:", round(r["precision"], 3), "recall:", round(r["recall"], 3))

    if args.set is not None:
        meta_path = MODEL_DIR / f"model-{MODEL_VERSION}.json"
        metadata = json.loads(meta_path.read_text())
        metadata["threshold"] = args.set
        metadata["val_at_threshold"] = confusion_at(y, scores, args.set)
        meta_path.write_text(json.dumps(metadata, indent=2))
        print("saved threshold", args.set, "to", meta_path)


if __name__ == "__main__":
    main()
