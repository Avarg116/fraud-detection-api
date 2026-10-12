"""Loads the trained model + metadata and turns transaction into a decision."""

import json
from pathlib import Path
import joblib
import pandas as pd
import sklearn

from fraud.api.schemas import Transaction


class FraudModel:
    def __init__(self, model_dir: Path, version: str) -> None:
        metadata = json.loads((model_dir / f"model-{version}.json").read_text())

        # joblib files are only reliable on the scikit-learn version they were saved with.
        if metadata["sklearn_version"] != sklearn.__version__:
            raise RuntimeError(
                f"model {version} was saved with scikit-learn {metadata['sklearn_version']}, "
                f"but {sklearn.__version__} is installed"
            )
        if "threshold" not in metadata:
            raise RuntimeError(
                f"model {version} has no threshold; run: python -m fraud.training.threshold --set <t>"
            )

        self.version: str = metadata["model_version"]
        self.threshold: float = metadata["threshold"]
        self.features: list[str] = metadata["features"]
        self._pipeline = joblib.load(model_dir / f"model-{version}.joblib")

    def predict(self, tx: Transaction) -> tuple[float, str]:
        
        values = tx.model_dump()
        row = pd.DataFrame([[values[name.lower()] for name in self.features]], columns=self.features)
        score = float(self._pipeline.predict_proba(row)[0, 1])
        decision = "FLAG" if score >= self.threshold else "APPROVE"
        return score, decision
