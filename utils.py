# utils.py
import logging
from flask import jsonify

def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        filename='edtech_api.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def validate_input(data: dict, required_fields: list) -> tuple:
    """Validate incoming JSON data."""
    missing = [field for field in required_fields if field not in data]
    if missing:
        return False, jsonify({"error": f"Missing fields: {missing}"}), 400
    return True, None, None
