import os
import re
import pandas as pd
import torch
import torch.nn as nn
import logging
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from model import LoanFraudModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATA_PATH = "data/loan.csv"
MODEL_DIR = "model"
MODEL_NAME = "fraud_model.pth"
SCALER_NAME = "scaler.pkl"
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

def extract_features(text):
    """Extract numeric features from text field with robust regex."""
    patterns = {
        "installment": r"Installment is (\d+\.?\d*)",
        "loan_amount": r"Loan Amount is (\d+\.?\d*)",
        "revolving_balance": r"Revolving Balance is (\d+\.?\d*)",
        "delinquency_2years": r"Delinquency In 2 years is (\d+\.?\d*)",
        "inquiries_6months": r"Inquiries In 6 Months is (\d+\.?\d*)",
        "mortgage_accounts": r"Mortgage Accounts is (\d+\.?\d*)",
        "open_accounts": r"Open Accounts is (\d+\.?\d*)",
        "revolving_utilization": r"Revolving Utilization Rate is (\d+\.?\d*)",
        "total_accounts": r"Total Accounts is (\d+\.?\d*)",
        "fico_range_low": r"Fico Range Low is (\d+\.?\d*)",
        "fico_range_high": r"Fico Range High is (\d+\.?\d*)",
        "annual_income": r"Annual Income is (\d+\.?\d*)",
    }
    
    features = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, str(text))
        features[key] = float(match.group(1)) if match else 0.0
    return features

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data file not found at {DATA_PATH}")
        return

    # Load and process data
    logger.info("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    logger.info("Extracting features...")
    features_list = df["text"].apply(extract_features).tolist()
    X_df = pd.DataFrame(features_list)
    
    # Target column: "good" -> 0, "bad" -> 1
    y = df["answer"].apply(lambda x: 1 if str(x).strip().lower() == "bad" else 0).values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_df.values, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
    
    # Initialize Model
    model = LoanFraudModel(input_size=X_train.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training Loop
    logger.info(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_t)
                test_loss = criterion(test_outputs, y_test_t)
                preds = (torch.sigmoid(test_outputs) > 0.5).float()
                acc = (preds == y_test_t).float().mean()
            logger.info(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}, Test Acc: {acc.item():.4f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        preds = (torch.sigmoid(test_outputs) > 0.5).float()
        final_acc = (preds == y_test_t).float().mean()
        logger.info(f"Final Test Accuracy: {final_acc.item():.4f}")

    # Save artifacts
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    scaler_path = os.path.join(MODEL_DIR, SCALER_NAME)
    
    torch.save(model.state_dict(), model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
        
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Scaler saved to {scaler_path}")

if __name__ == "__main__":
    train()