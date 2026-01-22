import os
import json
import numpy as np
import cv2
from labelme import utils

IMG_DIR = r"D:\project\annotations\images"
MASK_DIR = r"D:\project\annotations\masks"

os.makedirs(MASK_DIR, exist_ok=True)

print("Processing from:", IMG_DIR)

for file in os.listdir(IMG_DIR):
    if file.endswith(".json"):
        json_path = os.path.join(IMG_DIR, file)

        with open(json_path, "r") as f:
            data = json.load(f)

        # Handle Labelme images saved as imagePath (not base64)
        if data.get("imageData") is None:
            image_path = os.path.join(IMG_DIR, data["imagePath"])
            img = cv2.imread(image_path)
        else:
            img = utils.img_b64_to_arr(data["imageData"])

        lbl, _ = utils.shapes_to_label(
            img.shape,
            data["shapes"],
            label_name_to_value={"leaf": 1}
        )

        mask = (lbl * 255).astype(np.uint8)

        mask_name = file.replace(".json", ".png")
        cv2.imwrite(os.path.join(MASK_DIR, mask_name), mask)

print("Mask generation completed successfully.")
