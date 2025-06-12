import pandas as pd
import pickle
from typing import Dict

class NaiveModel:
    def __init__(self):
        self.means: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> None:
        """Calculate and store the mean for each column."""
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty")
        self.means = df.mean().to_dict()

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame divided by learned column means."""
        if not self.means:
            raise ValueError("Model has not been fitted yet")
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty")
        result = df.copy(deep=True)
        for column, mean in self.means.items():
            if column in result.columns and mean != 0:
                result[column] = result[column] / mean
        return result

    def save(self, path: str) -> None:
        """Serialize column means to disk using pickle."""
        with open(path, 'wb') as f:
            pickle.dump(self.means, f)

    def load(self, path: str) -> None:
        """Load column means from disk."""
        with open(path, 'rb') as f:
            self.means = pickle.load(f)

