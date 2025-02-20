# preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    """Preprocesses educational data for AI models."""
    def __init__(self):
        self.scaler = StandardScaler()

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from a CSV file."""
        return pd.read_csv(filepath)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare educational data."""
        df = df.dropna()  # Drop missing values (customize as needed)
        return df

    def scale_features(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """Scale numerical features."""
        df_scaled = df.copy()
        df_scaled[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        return df_scaled
