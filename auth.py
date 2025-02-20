# auth.py
import jwt
from flask import request, jsonify
from datetime import datetime, timedelta
from .config import Config

class AuthManager:
    """Handles user authentication using JWT."""
    SECRET_KEY = Config.SECRET_KEY

    @staticmethod
    def generate_token(user_id: str) -> str:
        """Generate a JWT token for a user."""
        payload = {
            "user_id": user_id,
            "exp": datetime.utcnow() + timedelta(hours=24)  # Token expires in 24 hours
        }
        return jwt.encode(payload, AuthManager.SECRET_KEY, algorithm="HS256")

    @staticmethod
    def verify_token(token: str) -> dict:
        """Verify a JWT token and return payload if valid."""
        try:
            return jwt.decode(token, AuthManager.SECRET_KEY, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return {"error": "Token expired"}
        except jwt.InvalidTokenError:
            return {"error": "Invalid token"}

    @staticmethod
    def require_auth(func):
        """Decorator to protect endpoints with authentication."""
        def wrapper(*args, **kwargs):
            token = request.headers.get("Authorization", "").replace("Bearer ", "")
            if not token:
                return jsonify({"error": "Token required"}), 401
            payload = AuthManager.verify_token(token)
            if "error" in payload:
                return jsonify(payload), 401
            request.user_id = payload["user_id"]
            return func(*args, **kwargs)
        wrapper.__name__ = func.__name__  # Preserve endpoint name
        return wrapper
