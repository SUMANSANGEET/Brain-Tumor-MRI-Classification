
from pathlib import Path
import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# Project paths
model_path = Path("artifacts/models/best_model.h5")
output_path = Path("artifacts/models/best_model.keras")

# Check that the original model exists
if not model_path.exists():
    raise FileNotFoundError(
        f"Model not found: {model_path.resolve()}"
    )

# Load the model without restoring training configuration
model = tf.keras.models.load_model(
    str(model_path),
    compile=False
)

print("\nModel loaded successfully.")
model.summary()

# Save in the modern Keras format
output_path.parent.mkdir(parents=True, exist_ok=True)
model.save(str(output_path))

print("\nModel converted successfully:")
print(output_path.resolve())
print("File size:", round(output_path.stat().st_size / (1024 * 1024), 2), "MB")

# Verify the converted model
reloaded_model = tf.keras.models.load_model(
    str(output_path),
    compile=False
)

print("\nConverted model verified successfully.")
print("Input shape:", reloaded_model.input_shape)
print("Output shape:", reloaded_model.output_shape)
