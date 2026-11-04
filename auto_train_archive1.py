"""
auto_train_archive1.py
-----------------------
Trains on the flat-image dataset at C:\\Users\\souti\\Downloads\\archive (1).

The dataset has 117 images across ~95 classes (1-2 images each), so a full
CNN from scratch would overfit immediately.  Strategy:
  1. Reorganise the flat folder into class sub-folders (temp directory).
  2. Use MobileNetV2 (ImageNet weights) as a frozen feature extractor.
  3. Add a small trainable head on top.
  4. Apply heavy augmentation to compensate for the tiny per-class count.
  5. Fine-tune the top layers of MobileNetV2 after the head is warm.

Saved to: traffic_classifier_archive1.keras
"""

import os
import re
import shutil
import tempfile

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SRC_DIR    = r'C:\Users\souti\Downloads\archive (1)'
SAVE_PATH  = 'traffic_classifier_archive1.keras'
IMG_SIZE   = (96, 96)   # MobileNetV2 minimum is 32; 96 gives better features
BATCH_SIZE = 8          # tiny dataset — small batch
EPOCHS_HEAD = 30        # train head with frozen base
EPOCHS_FINE = 20        # fine-tune top layers of MobileNetV2
SEED       = 42

# ---------------------------------------------------------------------------
# Step 1: Organise flat images into class sub-folders in a temp directory
# ---------------------------------------------------------------------------
def extract_label(filename: str) -> str:
    """Extract class label from filename prefix (before first '_' or extension)."""
    stem = re.sub(r'\.(jpeg|jpg|png)$', '', filename, flags=re.IGNORECASE)
    parts = stem.split('_')
    return parts[0]


print("Organising images into class folders...")
tmp_root = tempfile.mkdtemp(prefix='tsr_archive1_')

for fname in os.listdir(SRC_DIR):
    if not re.search(r'\.(jpeg|jpg|png)$', fname, re.IGNORECASE):
        continue
    label = extract_label(fname)
    class_dir = os.path.join(tmp_root, label)
    os.makedirs(class_dir, exist_ok=True)
    shutil.copy2(os.path.join(SRC_DIR, fname), os.path.join(class_dir, fname))

classes = sorted(os.listdir(tmp_root))
num_classes = len(classes)
total_images = sum(len(os.listdir(os.path.join(tmp_root, c))) for c in classes)
print(f"  {total_images} images organised into {num_classes} classes.")
print(f"  Temp dir: {tmp_root}")

# ---------------------------------------------------------------------------
# Step 2: Build tf.data datasets with heavy augmentation
# ---------------------------------------------------------------------------
print("\nBuilding datasets...")

# With so few images, use 80/20 split but warn if val set is tiny
train_ds = tf.keras.utils.image_dataset_from_directory(
    tmp_root,
    validation_split=0.2,
    subset='training',
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    tmp_root,
    validation_split=0.2,
    subset='validation',
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False,
)

print(f"  Classes found: {len(train_ds.class_names)}")

# Heavy augmentation pipeline — critical for tiny datasets
augment = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.20),
    layers.RandomTranslation(0.15, 0.15),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
], name='augmentation')

AUTOTUNE = tf.data.AUTOTUNE

def prepare_train(ds):
    return (
        ds
        .map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
        .cache()
        .shuffle(500)
        .prefetch(AUTOTUNE)
    )

train_ds = prepare_train(train_ds)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)

# ---------------------------------------------------------------------------
# Step 3: Build model — MobileNetV2 feature extractor + custom head
# ---------------------------------------------------------------------------
print("\nBuilding MobileNetV2 transfer-learning model...")

base_model = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights='imagenet',
)
base_model.trainable = False   # freeze during head training

inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
x = tf.keras.layers.Rescaling(1./127.5, offset=-1.0)(inputs)   # MobileNetV2 expects [-1, 1]
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs, name='mobilenetv2_tsr')
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
model.summary()

# ---------------------------------------------------------------------------
# Step 4: Phase 1 — Train head (base frozen)
# ---------------------------------------------------------------------------
callbacks_head = [
    ModelCheckpoint(SAVE_PATH, save_best_only=True, monitor='val_accuracy', verbose=1),
    EarlyStopping(patience=8, monitor='val_accuracy', restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=4, monitor='val_loss', verbose=1),
]

print(f"\nPhase 1: Training head ({EPOCHS_HEAD} epochs max, base frozen)...")
history1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, callbacks=callbacks_head)

# ---------------------------------------------------------------------------
# Step 5: Phase 2 — Fine-tune top 30 layers of MobileNetV2
# ---------------------------------------------------------------------------
print("\nPhase 2: Fine-tuning top layers of MobileNetV2...")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # much lower LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

callbacks_fine = [
    ModelCheckpoint(SAVE_PATH, save_best_only=True, monitor='val_accuracy', verbose=1),
    EarlyStopping(patience=8, monitor='val_accuracy', restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=4, monitor='val_loss', verbose=1),
]

history2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE, callbacks=callbacks_fine)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
all_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
best_val_acc = max(all_val_acc)
print(f"\nBest val accuracy: {best_val_acc:.4f}")
print(f"Model saved to '{SAVE_PATH}'")
print(f"\nNote: Only {total_images} images across {num_classes} classes were available.")
print("Consider collecting more images per class for better accuracy.")

# Clean up temp directory
shutil.rmtree(tmp_root, ignore_errors=True)

