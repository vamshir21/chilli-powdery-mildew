import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow as tf
from tensorflow.keras import backend as K

def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    )

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return 0.7 * bce + 0.3 * dice_loss(y_true, y_pred)


# --------------------
# CONFIG
# --------------------
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 30

TRAIN_IMG_DIR = r"D:\project\segmentation_dataset\train\images"
TRAIN_MASK_DIR = r"D:\project\segmentation_dataset\train\masks"
VAL_IMG_DIR   = r"D:\project\segmentation_dataset\val\images"
VAL_MASK_DIR  = r"D:\project\segmentation_dataset\val\masks"

# --------------------
# DATA LOADER
# --------------------
def load_image(img_path, mask_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    mask = (mask > 127).astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)

    return img, mask

def data_generator(img_dir, mask_dir, batch_size):
    img_files = sorted(os.listdir(img_dir))
    while True:
        for i in range(0, len(img_files), batch_size):
            batch_imgs = []
            batch_masks = []
            for f in img_files[i:i+batch_size]:
                img_path = os.path.join(img_dir, f)
                mask_path = os.path.join(mask_dir, f.replace(".jpg", ".png"))
                img, mask = load_image(img_path, mask_path)
                batch_imgs.append(img)
                batch_masks.append(mask)
            yield np.array(batch_imgs), np.array(batch_masks)

# --------------------
# U-NET MODEL
# --------------------
def unet_model():
    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))

    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(c3)

    u4 = layers.UpSampling2D()(c3)
    u4 = layers.Concatenate()([u4, c2])
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(u4)
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(c4)

    u5 = layers.UpSampling2D()(c4)
    u5 = layers.Concatenate()([u5, c1])
    c5 = layers.Conv2D(32, 3, activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(32, 3, activation='relu', padding='same')(c5)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c5)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss=bce_dice_loss,
        metrics=['accuracy']
    )

    return model

# --------------------
# TRAIN
# --------------------
train_gen = data_generator(TRAIN_IMG_DIR, TRAIN_MASK_DIR, BATCH_SIZE)
val_gen   = data_generator(VAL_IMG_DIR, VAL_MASK_DIR, BATCH_SIZE)

steps_per_epoch = len(os.listdir(TRAIN_IMG_DIR)) // BATCH_SIZE
val_steps = max(1, len(os.listdir(VAL_IMG_DIR)) // BATCH_SIZE)

model = unet_model()
model.summary()

model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=val_steps,
    epochs=EPOCHS
)

model.save("new_first_leaf_unet.h5")
print("Training complete. Model saved as leaf_unet.h5")
