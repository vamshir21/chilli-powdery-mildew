import os
import cv2
import numpy as np
import tensorflow as tf

# =====================
# CONFIG
# =====================
IMG_SIZE = 256

MODEL_PATH = r"D:\project\first_leaf_unet.h5"
IMG_DIR = r"D:\project\segmentation_dataset\all_images"
OUT_MASK_DIR = r"D:\project\segmentation_dataset\all_masks"

os.makedirs(OUT_MASK_DIR, exist_ok=True)

# =====================
# POST-PROCESSING
# =====================
def refine_mask(mask):
    mask = (mask > 0.5).astype(np.uint8) * 255
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

# =====================
# LOAD MODEL
# =====================
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# =====================
# RUN SEGMENTATION
# =====================
for img_name in os.listdir(IMG_DIR):
    if not img_name.endswith(".jpg"):
        continue

    img_path = os.path.join(IMG_DIR, img_name)
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    inp = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    inp = np.expand_dims(inp, axis=0)

    pred = model.predict(inp, verbose=0)[0, :, :, 0]
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    pred = refine_mask(pred)

    mask_name = img_name.replace(".jpg", ".png")
    cv2.imwrite(os.path.join(OUT_MASK_DIR, mask_name), pred)

print("✅ Segmentation masks generated for all images.")
