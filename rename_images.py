import os

# CHANGE THIS if dataset_final is elsewhere
BASE_DIR = "dataset_final"

CLASSES = ["healthy", "mild", "moderate", "severe"]

for cls in CLASSES:
    folder_path = os.path.join(BASE_DIR, cls)
    
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        continue

    images = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    for idx, img in enumerate(images, start=1):
        old_path = os.path.join(folder_path, img)
        ext = os.path.splitext(img)[1]
        new_name = f"{cls}_{idx:04d}{ext}"
        new_path = os.path.join(folder_path, new_name)

        os.rename(old_path, new_path)

    print(f"{cls}: renamed {len(images)} images")

print("Renaming completed successfully.")
