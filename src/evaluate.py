import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from config import (
    TEST_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    CLASS_NAMES,
    ARTIFACTS_DIR,
    REPORTS_DIR
)

from utils import plot_confusion_matrix


def load_test_dataset():

    return tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_names=CLASS_NAMES,
        shuffle=False
    )


def evaluate_model(model_name, test_ds):

    model_path = (
        ARTIFACTS_DIR /
        f"{model_name}_best.keras"
    )

    if not model_path.exists():

        print(
            f"Model not found: {model_path}"
        )

        return None

    model = tf.keras.models.load_model(
        model_path
    )

    y_true = np.concatenate([
        y.numpy()
        for _, y in test_ds
    ])

    probabilities = model.predict(
        test_ds,
        verbose=1
    )

    y_pred = np.argmax(
        probabilities,
        axis=1
    )

    metrics = {

        "accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "precision_macro":
            precision_score(
                y_true,
                y_pred,
                average="macro"
            ),

        "recall_macro":
            recall_score(
                y_true,
                y_pred,
                average="macro"
            ),

        "f1_macro":
            f1_score(
                y_true,
                y_pred,
                average="macro"
            )
    }

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES
    )

    print("\n" + "=" * 70)
    print(model_name.upper())
    print("=" * 70)

    print(json.dumps(
        metrics,
        indent=4
    ))

    print("\nClassification Report:")
    print(report)

    confusion_path = (
        REPORTS_DIR /
        f"{model_name}_confusion_matrix.png"
    )

    plot_confusion_matrix(
        y_true,
        y_pred,
        CLASS_NAMES,
        confusion_path
    )

    return metrics


def main():

    REPORTS_DIR.mkdir(exist_ok=True)

    test_ds = load_test_dataset()

    results = {}

    for model_name in [
        "cnn",
        "efficientnet"
    ]:

        result = evaluate_model(
            model_name,
            test_ds
        )

        if result:
            results[model_name] = result

    results_path = (
        REPORTS_DIR /
        "model_comparison.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print(
        f"\nComparison saved to: {results_path}"
    )


if __name__ == "__main__":
    main()