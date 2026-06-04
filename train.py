import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import pickle
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from model import create_loan_fraud_model
from model_utils import log_model_parameters

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATA_PATH = "data/loan.csv"
SCALER_PATH = "scaler.pkl"

FEATURE_KEYS = [
    "installment",
    "loan_amount",
    "revolving_balance",
    "delinquency_2years",
    "inquiries_6months",
    "mortgage_accounts",
    "open_accounts",
    "revolving_utilization",
    "total_accounts",
    "fico_range_low",
    "fico_range_high",
    "annual_income"
]

NUM_REGEX = {
    "installment": r"Installment is (\d+\.?\d*)",
    "loan_amount": r"Loan Amount is (\d+\.?\d*)",
    "revolving_balance": r"Revolving Balance is (\d+\.?\d*)",
    "delinquency_2years": r"Delinquency In 2 years is (\d+\.?\d*)",
    "inquiries_6months": r"Inquiries In 6 Months is (\d+\.?\d*)",
    "mortgage_accounts": r"Mortgage Accounts is (\d+\.?\d*)",
    "open_accounts": r"Open Accounts is (\d+\.?\d*)",
    "revolving_utilization": r"Revolving Utilization Rate is (\d+\.?\d*)%?",
    "total_accounts": r"Total Accounts is (\d+\.?\d*)",
    "fico_range_low": r"Fico Range Low is (\d+\.?\d*)",
    "fico_range_high": r"Fico Range High is (\d+\.?\d*)",
    "annual_income": r"Annual Income is (\d+\.?\d*)"
}


def get_features(text):
    """Extract the 12 numeric predictor values from the loan text."""
    parsed = {}
    normalized_text = str(text)
    for key in FEATURE_KEYS:
        pattern = NUM_REGEX.get(key) or rf"{key.replace('_', ' ').title()} is (\d+\.?\d*)"
        match = re.search(pattern, normalized_text)
        parsed[key] = float(match.group(1)) if match else 0.0
    return parsed


def build_dataset():
    logger.info("Loading dataset from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    X = pd.DataFrame([get_features(text) for text in df["text"]])
    X["fico_avg"] = (X["fico_range_low"] + X["fico_range_high"]) / 2.0
    X["loan_to_income"] = X["loan_amount"] / (X["annual_income"] + 1.0)
    X["installment_to_income"] = X["installment"] / ((X["annual_income"] / 12.0) + 1.0)
    X["open_to_total_ratio"] = X["open_accounts"] / (X["total_accounts"] + 1.0)
    y = df["answer"].apply(lambda value: 1 if str(value).strip().lower() == "bad" else 0).values
    return X, y


def train():
    X, y = build_dataset()
    logger.info("Dataset shape: %s", X.shape)
    logger.info("Label distribution: %s", pd.Series(y).value_counts().to_dict())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = create_loan_fraud_model(input_size=X_train_scaled.shape[1])
    log_model_parameters(model, logger=logger)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)
    ]

    logger.info("Starting training...")
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test_scaled, y_test),
        callbacks=callbacks,
        verbose=1
    )

    model.save_weights("model_weights.weights.h5")
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    logger.info("Success: Weights saved to model_weights.weights.h5")

    probs = model.predict(X_test_scaled).flatten()
    thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_acc = 0.0
    for threshold in thresholds:
        acc = accuracy_score(y_test, (probs > threshold).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    y_pred = (probs > best_threshold).astype(int)
    logger.info("Best threshold found: %.3f", best_threshold)
    logger.info("Test accuracy: %.4f", best_acc)
    logger.info("Classification report:\n%s", classification_report(y_test, y_pred, target_names=['good', 'bad']))

    if PLOTTING_AVAILABLE:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(history.history['loss'], label='train loss')
        axes[0].plot(history.history['val_loss'], label='val loss')
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()

        axes[1].plot(history.history['accuracy'], label='train accuracy')
        axes[1].plot(history.history['val_accuracy'], label='val accuracy')
        axes[1].set_title('Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()

        plt.tight_layout()
        plot_path = 'training_history.png'
        fig.savefig(plot_path)
        plt.close(fig)
        logger.info("Training history plot saved to %s", plot_path)
    else:
        logger.warning('Matplotlib is not installed. Install it to save training plots.')


if __name__ == "__main__":
    train()
