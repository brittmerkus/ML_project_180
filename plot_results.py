import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os

# Simulated training history (fake data)
epochs = 50
fake_history = {
    "accuracy": np.clip(np.linspace(0.5, 0.95, epochs) + np.random.uniform(-0.05, 0.05, epochs), 0, 1),
    "val_accuracy": np.clip(np.linspace(0.45, 0.90, epochs) + np.random.uniform(-0.05, 0.05, epochs), 0, 1),
    "loss": np.clip(np.linspace(1.5, 0.2, epochs) + np.random.uniform(-0.1, 0.1, epochs), 0, 3),
    "val_loss": np.clip(np.linspace(1.6, 0.3, epochs) + np.random.uniform(-0.1, 0.1, epochs), 0, 3),
}

# Fake final results for three models
models = ["CNN", "Neural Network", "Hybrid"]
# train_accuracy = [0.51, 0.83, 0.92]
# val_accuracy = [0.50, 0.78, 0.91]
# train_loss = [1.67, 0.30, 0.22]
# val_loss = [1.75, 0.32, 0.24]


def plot_barchart(train_acc, val_acc, train_loss, val_loss, subset_size):
    # Create output directory
    output_dir = 'output/graphs'
    os.makedirs(output_dir, exist_ok=True)

    # Bar chart for accuracy
    plt.figure(figsize=(6, 4))
    x = np.arange(len(models))
    width = 0.3

    plt.bar(x - width / 2, train_acc, width, label='Train Accuracy', color='blue')
    plt.bar(x + width / 2, val_acc, width, label='Validation Accuracy', color='orange')

    plt.xticks(x, models)
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.legend()
    plt.savefig(f'{output_dir}/graph_accuracy_subset_{subset_size}.png')
    plt.close()

    # Bar chart for loss
    plt.figure(figsize=(6, 4))

    plt.bar(x - width / 2, train_loss, width, label='Train Loss', color='blue')
    plt.bar(x + width / 2, val_loss, width, label='Validation Loss', color='orange')

    plt.xticks(x, models)
    plt.ylabel('Loss')
    plt.title('Model Loss Comparison')
    plt.legend()
    plt.savefig(f'{output_dir}/graph_loss_subset_{subset_size}.png')
    plt.close()


def plot_accuracy(history, subset_size):
    """Plots training and validation accuracy."""
    output_dir = 'output/graphs'
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(f'{output_dir}/graph_accuracy_history_subset_{subset_size}.png')
    plt.close()


def plot_loss(history, subset_size):
    """Plots training and validation loss."""
    output_dir = 'output/graphs'
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(history['loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(f'{output_dir}/graph_loss_history_subset_{subset_size}.png')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, subset_size):
    """Plots confusion matrix."""
    output_dir = 'output/graphs'
    os.makedirs(output_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    labels = ['Human', 'AI']

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(f'{output_dir}/graph_confusion_matrix_subset_{subset_size}.png')
    plt.close()


def plot_roc_curve(y_true, y_probs, subset_size):
    """Plots ROC curve and calculates AUC score."""
    output_dir = 'output/graphs'
    os.makedirs(output_dir, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, np.sort(y_probs))
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(f'{output_dir}/graph_roc_curve_subset_{subset_size}.png')
    plt.close()


# Modified test function
def run_plots(train_accuracy, val_accuracy, train_loss, val_loss, subset_size=1000):
    plot_barchart(train_accuracy, val_accuracy, train_loss, val_loss, subset_size)
    print('barcharts plotted')





# Run the plots
if __name__ == "__main__":
    run_plots(train_accuracy, val_accuracy, train_loss, val_loss, subset_size=1000)