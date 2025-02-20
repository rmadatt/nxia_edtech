# app.py
from flask import Flask, request, jsonify
from .auth import AuthManager
from .preprocessing import DataPreprocessor
from .recommendations import RecommendationEngine
from .utils import setup_logging, validate_input
from .config import Config

app = Flask(__name__)
app.config.from_object(Config)
logger = setup_logging()

# Initialize components
preprocessor = DataPreprocessor()
recommend_engine = RecommendationEngine()

# Load and train model (example data path)
data = preprocessor.load_data("sample_edu_data.csv")
data_scaled = preprocessor.scale_features(data, ["feature1", "feature2"])
recommend_engine.train(data_scaled, ["feature1", "feature2"], "target")

@app.route('/api/login', methods=['POST'])
def login():
    """User login endpoint."""
    data = request.json
    valid, error, status = validate_input(data, ["user_id", "password"])
    if not valid:
        return error, status
    # Simulate password check (replace with real auth logic)
    token = AuthManager.generate_token(data["user_id"])
    logger.info(f"User {data['user_id']} logged in")
    return jsonify({"token": token})

@app.route('/api/recommend', methods=['POST'])
@AuthManager.require_auth
def recommend():
    """Protected endpoint for content recommendations."""
    data = request.json
    valid, error, status = validate_input(data, ["features"])
    if not valid:
        return error, status
    prediction = recommend_engine.recommend(data["features"])
    logger.info(f"Recommendation made for user {request.user_id}")
    return jsonify({"recommendations": prediction})

if __name__ == "__main__":
    app.run(debug=True)
