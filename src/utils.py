import json
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def save_json(data, path):

    path = Path(path)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def plot_training_history(history, output_path):

    history_dict = history.history

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.plot(history_dict["accuracy"], label="Train Accuracy")
    plt.plot(history_dict["val_accuracy"], label="Validation Accuracy")

    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(history_dict["loss"], label="Train Loss")
    plt.plot(history_dict["val_loss"], label="Validation Loss")

    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    output_path
):

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 7))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45
    )

    plt.title("Brain Tumor MRI Classification - Confusion Matrix")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()


def save_classification_report(
    y_true,
    y_pred,
    class_names,
    output_path
):

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True
    )

    save_json(report, output_path)

    return report