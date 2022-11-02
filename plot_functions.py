import matplotlib.pyplot as plt


def plot_history(history, name):
    """Plot training and validation accuracy and loss."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
    ax1.plot(history.history["accuracy"], label="Training Accuracy")
    ax1.plot(history.history["val_accuracy"], label="Validation Accuracy")
    ax1.set_title("Training and Validation Accuracy")
    ax1.legend()
    ax2.plot(history.history["loss"], label="Training Loss")
    ax2.plot(history.history["val_loss"], label="Validation Loss")
    ax2.set_title(f"Training and Validation Loss for {name}")
    ax2.legend()
    plt.show()
