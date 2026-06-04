import tensorflow as tf
from tensorflow.keras import layers, models

def create_loan_fraud_model(input_size=12):
    """
    Creates a stronger Keras model for loan fraud classification.
    Uses batch normalization and dropout to improve generalization.
    """
    model = models.Sequential([
        layers.Input(shape=(input_size,)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.15),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model

