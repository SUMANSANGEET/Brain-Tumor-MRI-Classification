from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "Tumour"

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]

SPLITS = ["train", "valid", "test"]


def main():

    print("=" * 70)
    print("BRAIN TUMOR MRI DATASET SETUP")
    print("=" * 70)

    if not DATA_DIR.exists():
        print(f"\nERROR: Dataset directory not found:")
        print(DATA_DIR)
        sys.exit(1)

    total_images = 0

    for split in SPLITS:

        split_dir = DATA_DIR / split

        if not split_dir.exists():
            print(f"\nERROR: Missing split: {split}")
            continue

        print(f"\n[{split.upper()}]")

        split_total = 0

        for class_name in CLASS_NAMES:

            class_dir = split_dir / class_name

            if not class_dir.exists():
                print(f"  ❌ {class_name}: MISSING")
                continue

            images = [
                p for p in class_dir.iterdir()
                if p.suffix.lower() in
                [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
            ]

            count = len(images)
            split_total += count

            print(f"  {class_name:<15} {count:>6} images")

        print(f"  {'TOTAL':<15} {split_total:>6}")

        total_images += split_total

    print("\n" + "=" * 70)
    print(f"TOTAL DATASET IMAGES: {total_images}")
    print("=" * 70)


if __name__ == "__main__":
    main()