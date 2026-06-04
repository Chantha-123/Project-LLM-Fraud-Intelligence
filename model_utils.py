import os
import tensorflow as tf

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def count_model_parameters(model):
    """Return the total number of trainable parameters in a Keras model."""
    return model.count_params()


def log_model_parameters(model, logger=None):
    """Log or print the total parameter count for a Keras model."""
    total_params = count_model_parameters(model)
    message = f"Model summary: {total_params:,} total trainable parameters"
    if logger:
        logger.info(message)
    else:
        print(message)
    return total_params


def print_model_summary(model):
    """Print a Keras model summary and return trainable/non-trainable totals."""
    model.summary()
    trainable_params = int(sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]))
    non_trainable_params = int(sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]))
    total_params = trainable_params + non_trainable_params
    print(f"\nTotal params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {non_trainable_params:,}")
    return trainable_params, non_trainable_params, total_params


def generate_parameter_plot(model, filename='model_parameters.png'):
    """Generate a small bar chart image showing model parameter counts."""
    if not PLOTTING_AVAILABLE:
        raise RuntimeError('Matplotlib is not installed, cannot generate plot.')

    trainable_params = int(sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]))
    non_trainable_params = int(sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]))
    total_params = trainable_params + non_trainable_params

    labels = ['Trainable', 'Non-trainable', 'Total']
    values = [trainable_params, non_trainable_params, total_params]
    colors = ['#4C72B0', '#DD8452', '#55A868']

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title('Model Parameter Counts')
    ax.set_ylabel('Parameter Count')
    ax.set_ylim(0, total_params * 1.1)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.98, f'{value:,}',
                ha='center', va='top', color='white', fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    return filename


if __name__ == "__main__":
    import sys
    from model import create_loan_fraud_model

    input_size = 12
    if len(sys.argv) > 1:
        try:
            input_size = int(sys.argv[1])
        except ValueError:
            print(f"Invalid input size '{sys.argv[1]}', using default 12.")

    model = create_loan_fraud_model(input_size=input_size)
    print(f"Model: \"{model.name}\"")
    trainable_params, non_trainable_params, total_params = print_model_summary(model)
    if PLOTTING_AVAILABLE:
        image_path = generate_parameter_plot(model)
        print(f"Parameter plot saved to {image_path}")
    else:
        print('Matplotlib is not available. Install matplotlib to generate a parameter plot.')
