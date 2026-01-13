import cv2
import os

# INPUT & OUTPUT ROOTS
INPUT_ROOT = r"D:\project\dataset_split"
OUTPUT_ROOT = r"D:\project\dataset_preprocessed"

SPLITS = ["train", "val", "test"]
CLASSES = ["healthy", "mild", "moderate", "severe"]

# CLAHE configuration (balanced, not aggressive)
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

for split in SPLITS:
    for cls in CLASSES:
        input_dir = os.path.join(INPUT_ROOT, split, cls)
        output_dir = os.path.join(OUTPUT_ROOT, split, cls)

        os.makedirs(output_dir, exist_ok=True)

        for img_name in os.listdir(input_dir):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(input_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Skipping unreadable file: {img_path}")
                continue

            # Convert to LAB
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE to L channel
            l_clahe = clahe.apply(l)

            # Merge channels back
            lab_clahe = cv2.merge((l_clahe, a, b))
            final_img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

            # Save
            cv2.imwrite(
                os.path.join(output_dir, img_name),
                final_img
            )

        print(f"Processed {split}/{cls}")

print("CLAHE preprocessing completed successfully.")
