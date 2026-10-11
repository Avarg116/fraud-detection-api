"""Compare logistic regression and LightGBM."""

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score
from fraud.training.data import FEATURES, TARGET
from fraud.training.train import build_model, load_train_val

LGBM_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "num_leaves": 15,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}
N_BOOTSTRAP = 1000


def main() -> None:
    train, val = load_train_val()
    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET].to_numpy()

    lr = build_model().fit(X_train, y_train)
    s_lr = lr.predict_proba(X_val)[:, 1]

    gbm = lgb.LGBMClassifier(**LGBM_PARAMS)
    gbm.fit(
        X_train,
        y_train,
        eval_X=(X_val,),
        eval_y=(y_val,),
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
    )
    s_gbm = gbm.predict_proba(X_val)[:, 1]

    print("validation frauds:", y_val.sum())
    print("logistic regression PR-AUC:", round(average_precision_score(y_val, s_lr), 4))
    print("LightGBM PR-AUC:", round(average_precision_score(y_val, s_gbm), 4),
          "(best iteration", gbm.best_iteration_, ")")

    # Resample validation rows with replacement and rescore.
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(y_val), len(y_val))
        diffs.append(average_precision_score(y_val[idx], s_gbm[idx])
                     - average_precision_score(y_val[idx], s_lr[idx]))
    diffs = np.array(diffs)
    low, high = np.percentile(diffs, [2.5, 97.5])
    print("\nbootstrap, LightGBM - LR:")
    print("mean diff:", round(diffs.mean(), 4))
    print("95% interval:", round(low, 4), round(high, 4))
    print("LightGBM better in", round(np.mean(diffs > 0) * 100), "% of resamples")


if __name__ == "__main__":
    main()
