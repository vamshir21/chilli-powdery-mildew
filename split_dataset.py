import os
import shutil
import random

SOURCE_DIR = r"D:\project\Dataset_final"
DEST_DIR = r"D:\project\Dataset_split"

CLASSES = ["healthy", "mild", "moderate", "severe"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)  # reproducibility

for cls in CLASSES:
    src_cls_dir = os.path.join(SOURCE_DIR, cls)
    images = [f for f in os.listdir(src_cls_dir)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in splits.items():
        dst_dir = os.path.join(DEST_DIR, split, cls)
        os.makedirs(dst_dir, exist_ok=True)

        for file in files:
            shutil.copy(
                os.path.join(src_cls_dir, file),
                os.path.join(dst_dir, file)
            )

    print(f"{cls}: {len(splits['train'])} train, "
          f"{len(splits['val'])} val, "
          f"{len(splits['test'])} test")

print("Dataset splitting completed successfully.")
