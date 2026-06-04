import os
import re
import pandas as pd
import tensorflow as tf
import logging
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from model import create_loan_fraud_model

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATA_PATH = "data/loan.csv"
MODEL_PATH = "fraud_model.keras"
SCALER_PATH = "scaler.pkl"



def get_features(text):
    """Simple feature extraction using regex."""
    keys = ["installment", "loan_amount", "revolving_balance", "delinquency_2years", 
            "inquiries_6months", "mortgage_accounts", "open_accounts", 
            "revolving_utilization", "total_accounts", "fico_range_low", 
            "fico_range_high", "annual_income"]
    
    features = {}
    for key in keys:
        pattern = rf"{key.replace('_', ' ').title()} is (\d+\.?\d*)"
        match = re.search(pattern, str(text))
        features[key] = float(match.group(1)) if match else 0.0
    return features

def train():
    os.makedirs("model", exist_ok=True)
    
    # 1. Load Data
    logger.info("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # 2. Extract Features & Labels
    logger.info("Processing features...")
    X = pd.DataFrame([get_features(t) for t in df["text"]]).values
    y = df["answer"].apply(lambda x: 1 if str(x).lower().strip() == "bad" else 0).values
    
    # 3. Split & Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 4. Create Model
    model = create_loan_fraud_model(input_size=X_train.shape[1])
    
    # 5. Train Model
    logger.info("Starting training...")
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # 6. Save Weights and Scaler
    model.save_weights("model_weights.weights.h5")
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
        
    logger.info("✅ Success: Weights saved to model_weights.weights.h5")



if __name__ == "__main__":
    train()