from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import pickle
import numpy as np
import logging
from model import LoanFraudModel
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Model configuration
MODEL_PATH = "model/fraud_model.pth"
SCALER_PATH = "model/scaler.pkl"
INPUT_SIZE = 12

class FraudPredictor:
    def __init__(self, model_path, scaler_path, input_size):
        self.model = LoanFraudModel(input_size)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"Model file not found at {model_path}")
        
        self.model.eval()
        
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            logger.info(f"Scaler loaded from {scaler_path}")
        else:
            self.scaler = None
            logger.warning(f"Scaler file not found at {scaler_path}")

    def predict(self, features):
        if self.scaler is None:
            raise Exception("Scaler not initialized")
        
        data = np.array(features).reshape(1, -1)
        data = self.scaler.transform(data)
        x = torch.tensor(data, dtype=torch.float32)

        with torch.no_grad():
            prob = torch.sigmoid(self.model(x)).item()
        
        return prob

# Initialize predictor
try:
    predictor = FraudPredictor(MODEL_PATH, SCALER_PATH, INPUT_SIZE)
except Exception as e:
    logger.error(f"Failed to initialize predictor: {e}")
    predictor = None

@app.route("/", methods=["GET"])
@app.route("/index.html", methods=["GET"])
def index():
    try:
        with open("index.html", "r") as f:
            return f.read()
    except Exception as e:
        return f"Error loading index.html: {e}", 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": predictor is not None})

@app.route("/predict", methods=["POST"])
def predict():
    if predictor is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.json.get("features")
        if not data or len(data) != INPUT_SIZE:
            return jsonify({"error": f"Invalid input. Expected {INPUT_SIZE} features."}), 400
        
        prob = predictor.predict(data)
        
        # Determine risk level and recommendation
        risk_level = "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low"
        recommendation = "Reject" if prob > 0.7 else "Manual Review" if prob > 0.3 else "Approve"

        result = {
            "result": "Fraud" if prob > 0.5 else "Safe",
            "probability": prob,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "status": "success"
        }
        
        logger.info(f"Prediction made: {result['result']} (prob: {prob:.4f})")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

