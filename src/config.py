from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset
DATA_DIR = PROJECT_ROOT / "data" / "Tumour"

TRAIN_DIR = DATA_DIR / "train"
VALID_DIR = DATA_DIR / "valid"
TEST_DIR = DATA_DIR / "test"

# Output directories
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Image configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Classification
CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]

NUM_CLASSES = len(CLASS_NAMES)

# Reproducibility
RANDOM_SEED = 42