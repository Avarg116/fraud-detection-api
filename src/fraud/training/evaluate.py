# Final eval on the test set

import json

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score

from fraud.training.data import FEATURES, TARGET, load, time_split
from fraud.training.threshold import confusion_at
from fraud.training.train import MODEL_DIR, MODEL_VERSION


def main() -> None:
    train, _val, test = time_split(load())

    # saved model and its threshold
    model = joblib.load(MODEL_DIR / f"model-{MODEL_VERSION}.joblib")
    metadata = json.loads((MODEL_DIR / f"model-{MODEL_VERSION}.json").read_text())
    threshold = metadata["threshold"]

    y = test[TARGET].to_numpy()
    scores = model.predict_proba(test[FEATURES])[:, 1]

    # baseline
    dummy = DummyClassifier(strategy="prior").fit(train[FEATURES], train[TARGET])
    dummy_scores = dummy.predict_proba(test[FEATURES])[:, 1]

    print("test set:", len(y), "transactions,", y.sum(), "frauds")
    print("model PR-AUC:", round(average_precision_score(y, scores), 4))
    print("baseline PR-AUC:", round(average_precision_score(y, dummy_scores), 4))

    r = confusion_at(y, scores, threshold)
    print("at threshold", threshold, "flagged:", r["flagged"], "caught:", r["caught"],
          "missed:", r["missed"], "false alarms:", r["false_alarms"],
          "precision:", round(r["precision"], 3), "recall:", round(r["recall"], 3))


if __name__ == "__main__":
    main()
