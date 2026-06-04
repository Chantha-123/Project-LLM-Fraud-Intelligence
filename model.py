import tensorflow as tf
from tensorflow.keras import layers, models

def create_loan_fraud_model(input_size=12):
    """
    Creates a simple Neural Network using Keras.
    Structure: Input (12) -> Hidden (64) -> Hidden (32) -> Output (1)
    """
    model = models.Sequential([
        layers.Input(shape=(input_size,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid') # Sigmoid used for binary classification (0 or 1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

