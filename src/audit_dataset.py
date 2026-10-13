from pathlib import Path
from PIL import Image
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "Tumour"
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(exist_ok=True)

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def audit_dataset():

    rows = []

    for split in ["train", "valid", "test"]:

        split_dir = DATA_DIR / split

        if not split_dir.exists():
            continue

        for class_dir in sorted(split_dir.iterdir()):

            if not class_dir.is_dir():
                continue

            class_name = class_dir.name

            for image_path in class_dir.rglob("*"):

                if image_path.suffix.lower() not in VALID_EXTENSIONS:
                    continue

                try:

                    with Image.open(image_path) as img:

                        width, height = img.size
                        mode = img.mode

                    rows.append({
                        "split": split,
                        "class": class_name,
                        "file": str(image_path),
                        "width": width,
                        "height": height,
                        "mode": mode,
                        "valid": True
                    })

                except Exception as e:

                    rows.append({
                        "split": split,
                        "class": class_name,
                        "file": str(image_path),
                        "width": None,
                        "height": None,
                        "mode": None,
                        "valid": False,
                        "error": str(e)
                    })

    df = pd.DataFrame(rows)

    output = REPORT_DIR / "dataset_audit.csv"
    df.to_csv(output, index=False)

    print("\nDataset audit completed.")
    print(f"Records: {len(df):,}")
    print(f"Valid images: {df['valid'].sum():,}")
    print(f"Invalid images: {(~df['valid']).sum():,}")

    print("\nClass distribution:")
    print(
        df.groupby(["split", "class"])
        .size()
        .reset_index(name="images")
        .to_string(index=False)
    )

    print(f"\nSaved report:")
    print(output)


if __name__ == "__main__":
    audit_dataset()