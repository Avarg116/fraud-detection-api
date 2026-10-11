
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from fraud.training.data import FEATURES, TARGET, load, time_split

MODEL_VERSION = "v1"
MODEL_DIR = Path("models")


def build_model() -> Pipeline:
    # class_weight="balanced" makes each fraud count ~540x more than a normal transaction
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    )


def load_train_val():
    train, val, _test = time_split(load())  # test until the final eval
    return train.drop_duplicates(), val


def main() -> None:
    train, val = load_train_val()
    print(f"train: {len(train):,} rows / {train[TARGET].sum()} frauds")

    model = build_model()
    model.fit(train[FEATURES], train[TARGET])

    val_scores = model.predict_proba(val[FEATURES])[:, 1]
    val_pr_auc = average_precision_score(val[TARGET], val_scores)
    print(f"validation PR-AUC: {val_pr_auc:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / f"model-{MODEL_VERSION}.joblib"
    joblib.dump(model, model_path)

    metadata = {
        "model_version": MODEL_VERSION,
        "model_type": "logistic_regression",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "features": FEATURES,
        "train_rows": len(train),
        "train_frauds": int(train[TARGET].sum()),
        "val_pr_auc": round(val_pr_auc, 4),
    }
    (MODEL_DIR / f"model-{MODEL_VERSION}.json").write_text(json.dumps(metadata, indent=2))
    print("saved model to", model_path)


if __name__ == "__main__":
    main()
