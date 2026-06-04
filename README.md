---
title: Loan Fraud Detection System
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# GuardLink.ai | Loan Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-black.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Loan Fraud Detection System is a machine learning application for classifying loan applications as `good` or `bad` using a TensorFlow/Keras model and a Flask inference API.

## What this repository contains

- `train.py` — training pipeline with feature extraction, scaling, model training, evaluation, and history plotting.
- `model.py` — Keras model definition with batch normalization and dropout.
- `model_utils.py` — reusable parameter-counting utilities and runnable model summary helper.
- `app.py` — Flask API service that applies the same feature engineering used during training.
- `data/loan.csv` — raw loan application dataset.
- `index.html` — simple frontend interface for manual predictions.

## Installation & Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python3 train.py
```

This will:
- load and parse the dataset
- create derived features such as `fico_avg`, `loan_to_income`, `installment_to_income`, and `open_to_total_ratio`
- train the model with early stopping and learning rate reduction
- save model weights to `model_weights.weights.h5`
- save standard scaler state to `scaler.pkl`
- generate `training_history.png` if `matplotlib` is installed

### 3. Inspect model parameters

```bash
python3 model_utils.py
```

### 4. Run the API service

```bash
python3 app.py
```

The Flask API accepts the same 12 loan feature values and applies the training-time preprocessing before returning a fraud score.

## What was improved

- stronger Keras model architecture with dropout and batch normalization
- derived features for improved numeric signal
- validation-based threshold search for test predictions
- training plot support in `training_history.png`
- separate model utilities in `model_utils.py`

## Prediction features

The training pipeline currently uses 12 direct loan inputs plus derived metrics:

- `installment`
- `loan_amount`
- `revolving_balance`
- `delinquency_2years`
- `inquiries_6months`
- `mortgage_accounts`
- `open_accounts`
- `revolving_utilization`
- `total_accounts`
- `fico_range_low`
- `fico_range_high`
- `annual_income`

Derived features:
- `fico_avg`
- `loan_to_income`
- `installment_to_income`
- `open_to_total_ratio`

## Notes

- The dataset is imbalanced: `good` loan examples dominate `bad` examples.
- Overall accuracy is a rough baseline; evaluating recall and precision for the `bad` class is essential for fraud detection.

## Optional Docker startup

```bash
docker-compose up --build
```

This project includes a `docker-compose.yml` file for containerized deployment.

---
© 2026 GuardLink Enterprise Solutions. All rights reserved.