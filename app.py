from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pickle
import numpy as np
import logging
import os
from model import create_loan_fraud_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Paths 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class FraudPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.last_error = "No model weights found"
        
        search_dirs = [os.getcwd(), BASE_DIR, "/app", "/"]
        for s_dir in search_dirs:
            if self.model and self.scaler: break
            try:
                if not os.path.exists(s_dir): continue
                files = os.listdir(s_dir)
                for f in files:
                    # 1. Load weights (the intelligent numbers)
                    if f.endswith("model_weights.weights.h5") and self.model is None:
                        try:
                            # Re-build architecture exactly as in training
                            self.model = create_loan_fraud_model(input_size=12)
                            self.model.load_weights(os.path.join(s_dir, f))
                            logger.info(f"✅ WEIGHTS LOADED FROM {f}")
                        except Exception as e:
                            self.last_error = f"Error loading weights: {str(e)}"
                            logger.error(self.last_error)

                    
                    # 2. Load scaler
                    if f.endswith(".pkl") and self.scaler is None:
                        try:
                            with open(os.path.join(s_dir, f), "rb") as file:
                                self.scaler = pickle.load(file)
                            logger.info(f"✅ SCALER LOADED FROM {f}")
                        except: pass
            except: continue

    def predict(self, features):
        data = np.array(features).reshape(1, -1)
        data = self.scaler.transform(data)
        prob = self.model.predict(data)[0][0]
        return float(prob)

predictor = FraudPredictor()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "model_loaded": predictor.model is not None,
        "error": predictor.last_error
    })

@app.route("/", methods=["GET"])
@app.route("/index.html", methods=["GET"])
def index():
    try:
        with open("index.html", "r") as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}", 500

@app.route("/predict", methods=["POST"])
def predict():
    if not predictor.model:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.json.get("features")
        prob = predictor.predict(data)
        
        result = "Fraud" if prob > 0.5 else "Safe"
        risk_level = "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low"
        recommendation = "Reject" if prob > 0.7 else "Manual Review" if prob > 0.3 else "Approve"

        return jsonify({
            "result": result,
            "probability": prob,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # REQUIRED: Use port 7860 for Hugging Face
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port, debug=False)
