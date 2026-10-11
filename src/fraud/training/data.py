"""Load the dataset and split it by time."""

from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw/creditcard.csv")
TARGET = "Class"
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15


def load(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path).sort_values("Time", kind="stable").reset_index(drop=True)


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    #Split into train/validation/test by time
    train_end = df["Time"].quantile(TRAIN_FRAC)
    val_end = df["Time"].quantile(TRAIN_FRAC + VAL_FRAC)

    train = df[df["Time"] < train_end]
    val = df[(df["Time"] >= train_end) & (df["Time"] < val_end)]
    test = df[df["Time"] >= val_end]
    return train, val, test
