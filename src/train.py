import argparse
from pathlib import Path

import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models

EfficientNetB0 = tf.keras.applications.EfficientNetB0
EarlyStopping = tf.keras.callbacks.EarlyStopping
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau

# Use the TensorFlow namespace already imported above to avoid IDE import-resolution warnings.

from config import (
    TRAIN_DIR,
    VALID_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    NUM_CLASSES,
    CLASS_NAMES,
    ARTIFACTS_DIR,
    REPORTS_DIR,
    RANDOM_SEED,
)

from utils import plot_training_history


def load_datasets():

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_names=CLASS_NAMES,
        shuffle=True,
        seed=RANDOM_SEED
    )

    valid_ds = tf.keras.utils.image_dataset_from_directory(
        VALID_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_names=CLASS_NAMES,
        shuffle=False
    )

    normalization = layers.Rescaling(1.0 / 255)

    train_ds = train_ds.map(
        lambda x, y: (normalization(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    valid_ds = valid_ds.map(
        lambda x, y: (normalization(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, valid_ds


def build_cnn():

    model = models.Sequential([

        layers.Input(shape=(*IMG_SIZE, 3)),

        layers.Conv2D(
            32,
            3,
            activation="relu"
        ),

        layers.MaxPooling2D(),

        layers.Conv2D(
            64,
            3,
            activation="relu"
        ),

        layers.MaxPooling2D(),

        layers.Conv2D(
            128,
            3,
            activation="relu"
        ),

        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation="relu"),

        layers.Dropout(0.4),

        layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    ])

    return model


def build_efficientnet():

    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3)
    )

    base_model.trainable = False

    inputs = layers.Input(
        shape=(*IMG_SIZE, 3)
    )

    x = base_model(inputs, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = models.Model(
        inputs,
        outputs
    )

    return model


def train(model_name, epochs):

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    train_ds, valid_ds = load_datasets()

    if model_name == "cnn":
        model = build_cnn()

    elif model_name == "efficientnet":
        model = build_efficientnet()

    else:
        raise ValueError(
            "model must be cnn or efficientnet"
        )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    model_path = (
        ARTIFACTS_DIR /
        f"{model_name}_best.keras"
    )

    callbacks = [

        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-6
        ),

        ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            save_best_only=True
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    history_path = (
        REPORTS_DIR /
        f"{model_name}_history.png"
    )

    plot_training_history(
        history,
        history_path
    )

    print("\nTraining completed.")
    print(f"Model saved to: {model_path}")
    print(f"Training plot: {history_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=[
            "cnn",
            "efficientnet"
        ],
        required=True
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=25
    )

    args = parser.parse_args()

    train(
        args.model,
        args.epochs
    )