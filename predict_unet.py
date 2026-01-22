import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# =====================
# CONFIG
# =====================
IMG_SIZE = 256
NUM_IMAGES = 5

MODEL_PATH = r"D:\project\new_first_leaf_unet.h5"
IMG_DIR = r"D:\project\segmentation_dataset\val\images"

# =====================
# POST-PROCESSING
# =====================
def refine_mask(mask):
    """
    mask: binary mask (0 or 1) at original image size
    returns: cleaned binary mask (0 or 255)
    """
    mask = (mask > 0).astype(np.uint8) * 255
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

# =====================
# LOAD MODEL (NO CUSTOM LOSS NEEDED)
# =====================
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully.")

# =====================
# LOAD IMAGES
# =====================
image_files = sorted(os.listdir(IMG_DIR))[:NUM_IMAGES]

for idx, img_name in enumerate(image_files, 1):
    img_path = os.path.join(IMG_DIR, img_name)

    # Read original image
    orig = cv2.imread(img_path)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    h, w, _ = orig.shape

    # Preprocess for model
    img = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict (256x256)
    pred_small = model.predict(img, verbose=0)[0, :, :, 0]
    pred_small = (pred_small > 0.5).astype(np.uint8)

    # Resize to original size
    pred = cv2.resize(
        pred_small,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    # ✅ APPLY POST-PROCESSING HERE
    pred = refine_mask(pred)

    # Overlay
    overlay = orig.copy()
    overlay[pred == 255] = [0, 255, 0]

    # =====================
    # DISPLAY RESULTS
    # =====================
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title(f"Image {idx}")
    plt.imshow(orig)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Predicted Mask")
    plt.imshow(pred, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.show()
