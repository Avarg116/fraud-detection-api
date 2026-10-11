"""Baseline scores the real model has to beat."""

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from fraud.training.data import FEATURES, TARGET, load, time_split


def main() -> None:
    train, val, test = time_split(load())

    print("time split:")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(name, len(part), "rows,", part[TARGET].sum(), "frauds")

    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]

    models = {
        "dummy (prior)": DummyClassifier(strategy="prior"),
        "logistic regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000),
        ),
    }

    print("\nvalidation scores:")
    for name, model in models.items():
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, scores)
        acc = accuracy_score(y_val, model.predict(X_val))
        print(name, "PR-AUC:", round(pr_auc, 4), "accuracy:", round(acc, 4))


if __name__ == "__main__":
    main()
