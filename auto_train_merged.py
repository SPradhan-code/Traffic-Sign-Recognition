"""
auto_train_merged.py
--------------------
Merges all three traffic-sign datasets directly into a unified pipeline
WITHOUT copying files, and trains a high-accuracy, fast CNN model.

Datasets:
  1. GTSRB (German)    : 39,209 images, 43 classes
  2. Indian Dataset    : 13,971 images, 58 classes
  3. Archive (1)       : 117 images, 95 classes

Total: 53,297 images across 190 unique classes.
Output:
  - merged_model.keras
  - merged_labels.json
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import glob
import re
import csv
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GTSRB_TRAIN   = r'D:\traffic-sign-recognition\traffic-sign-recognition\data\raw\Train'
INDIAN_DIR    = r'C:\Users\souti\Downloads\archive\Indian-Traffic Sign-Dataset\Images'
INDIAN_CSV    = r'C:\Users\souti\Downloads\archive\Indian-Traffic Sign-Dataset\traffic_sign.csv'
ARCHIVE1_DIR  = r'C:\Users\souti\Downloads\archive (1)'

SAVE_PATH     = 'merged_model.keras'
LABELS_PATH   = 'merged_labels.json'

IMG_SIZE      = (32, 32)
BATCH_SIZE    = 64
EPOCHS        = 20
SEED          = 42

GTSRB_NAMES = [
    "Speed_limit_20", "Speed_limit_30", "Speed_limit_50", "Speed_limit_60",
    "Speed_limit_70", "Speed_limit_80", "End_speed_limit_80", "Speed_limit_100",
    "Speed_limit_120", "No_passing", "No_passing_3t5", "Right_of_way",
    "Priority_road", "Yield", "Stop", "No_vehicles", "No_vehicles_3t5",
    "No_entry", "General_caution", "Curve_left", "Curve_right",
    "Double_curve", "Bumpy_road", "Slippery_road", "Road_narrows",
    "Road_work", "Traffic_signals", "Pedestrians", "Children_crossing",
    "Bicycles_crossing", "Beware_ice_snow", "Wild_animals", "End_all_limits",
    "Turn_right_ahead", "Turn_left_ahead", "Ahead_only", "Go_straight_or_right",
    "Go_straight_or_left", "Keep_right", "Keep_left", "Roundabout",
    "End_no_passing", "End_no_passing_3t5"
]

def sanitize(name: str) -> str:
    return re.sub(r'[^\w]+', '_', name).strip('_')

print("=" * 60, flush=True)
print("STEP 1: Scanning datasets (Zero-copy in-memory indexing)...", flush=True)
print("=" * 60, flush=True)

samples = []

# 1. GTSRB
for cid_str in sorted(os.listdir(GTSRB_TRAIN)):
    try:
        cid = int(cid_str)
        name = GTSRB_NAMES[cid]
        label = f"gtsrb__{sanitize(name)}"
        for p in glob.glob(os.path.join(GTSRB_TRAIN, cid_str, "*.png")):
            samples.append((p, label))
    except (ValueError, IndexError):
        pass
print(f"  GTSRB images indexed: {len(samples):,}", flush=True)
gtsrb_count = len(samples)

# 2. Indian
indian_map = {}
if os.path.exists(INDIAN_CSV):
    with open(INDIAN_CSV, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            indian_map[int(row['ClassId'])] = row['Name']

indian_prev = len(samples)
for cid_str in sorted(os.listdir(INDIAN_DIR)):
    try:
        cid = int(cid_str)
        name = indian_map.get(cid, f"class_{cid}")
        label = f"indian__{sanitize(name)}"
        for p in glob.glob(os.path.join(INDIAN_DIR, cid_str, "*.*")):
            if p.lower().endswith(('.png', '.jpg', '.jpeg')):
                samples.append((p, label))
    except ValueError:
        pass
print(f"  Indian dataset images indexed: {len(samples) - indian_prev:,}", flush=True)

# 3. Archive (1)
a1_prev = len(samples)
for p in glob.glob(os.path.join(ARCHIVE1_DIR, "*.*")):
    if p.lower().endswith(('.png', '.jpg', '.jpeg')):
        fname = os.path.basename(p)
        stem = re.sub(r'\.(jpeg|jpg|png)$', '', fname, flags=re.IGNORECASE)
        prefix = stem.split('_')[0]
        label = f"a1__{sanitize(prefix)}"
        samples.append((p, label))
print(f"  Archive (1) images indexed: {len(samples) - a1_prev:,}", flush=True)

total_images = len(samples)
all_classes = sorted(list(set(s[1] for s in samples)))
num_classes = len(all_classes)
print(f"\nTotal images: {total_images:,} across {num_classes} unified classes.", flush=True)

# Save labels JSON
class_to_idx = {name: idx for idx, name in enumerate(all_classes)}
idx_to_class = {idx: name for idx, name in enumerate(all_classes)}
with open(LABELS_PATH, 'w', encoding='utf-8') as f:
    json.dump(idx_to_class, f, indent=2)
print(f"Saved class labels map to '{LABELS_PATH}'.", flush=True)

# ---------------------------------------------------------------------------
# Train / Validation Split
# ---------------------------------------------------------------------------
print("\n" + "=" * 60, flush=True)
print("STEP 2: Stratified Train / Validation Splitting...", flush=True)
print("=" * 60, flush=True)

# Separate single-sample classes from multi-sample classes for clean split
from collections import Counter
counts = Counter(s[1] for s in samples)

train_paths, train_labels = [], []
val_paths, val_labels = [], []

multi_samples = []
for p, lbl in samples:
    if counts[lbl] < 2:
        train_paths.append(p)
        train_labels.append(class_to_idx[lbl])
    else:
        multi_samples.append((p, lbl))

multi_paths = [s[0] for s in multi_samples]
multi_lbls  = [class_to_idx[s[1]] for s in multi_samples]

m_tr_p, m_val_p, m_tr_l, m_val_l = train_test_split(
    multi_paths, multi_lbls,
    test_size=0.15,
    random_state=SEED,
    stratify=multi_lbls
)

train_paths.extend(m_tr_p)
train_labels.extend(m_tr_l)
val_paths.extend(m_val_p)
val_labels.extend(m_val_l)

print(f"  Train set: {len(train_paths):,} images", flush=True)
print(f"  Val set  : {len(val_paths):,} images", flush=True)

# ---------------------------------------------------------------------------
# tf.data Pipeline
# ---------------------------------------------------------------------------
def load_and_preprocess(path, label):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    one_hot = tf.one_hot(label, depth=num_classes)
    return img, one_hot

AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.shuffle(buffer_size=10000, seed=SEED)
train_ds = train_ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)

# Data augmentation layer inside tf.data
augment = tf.keras.Sequential([
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomTranslation(0.08, 0.08),
])
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------
print("\n" + "=" * 60, flush=True)
print("STEP 3: Compiling CNN Model...", flush=True)
print("=" * 60, flush=True)

inputs = layers.Input(shape=(*IMG_SIZE, 3))
x = layers.Rescaling(1.0 / 255)(inputs)

# Block 1
x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.25)(x)

# Block 2
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.25)(x)

# Block 3
x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.25)(x)

# Head
x = layers.Flatten()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs=inputs, outputs=outputs, name='unified_tsr_cnn')
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
print("\n" + "=" * 60, flush=True)
print("STEP 4: Training Unified Model...", flush=True)
print("=" * 60, flush=True)

callbacks = [
    ModelCheckpoint(SAVE_PATH, save_best_only=True, monitor='val_accuracy', verbose=1),
    EarlyStopping(patience=5, monitor='val_accuracy', restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=2, monitor='val_loss', min_lr=1e-6, verbose=1),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

best_val = max(history.history['val_accuracy'])
print("\n" + "=" * 60, flush=True)
print(f"🎉 Training Finished! Best Validation Accuracy: {best_val:.4%}", flush=True)
print(f"📁 Model saved to: '{SAVE_PATH}'", flush=True)
print(f"🏷️  Labels saved to: '{LABELS_PATH}'", flush=True)
print("=" * 60, flush=True)
