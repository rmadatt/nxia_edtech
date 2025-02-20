# recommendations.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

class RecommendationEngine:
    """Generates AI recommendations for educational content."""
    def __init__(self):
        self.model = RandomForestClassifier()

    def train(self, data: pd.DataFrame, features: list, target: str):
        """Train the recommendation model."""
        X = data[features]
        y = data[target]
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

    def recommend(self, user_features: list) -> list:
        """Predict recommended content for a user."""
        prediction = self.model.predict([user_features])
        return prediction.tolist()
