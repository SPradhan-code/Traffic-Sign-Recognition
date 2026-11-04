import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import csv
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# -- Paths --------------------------------------------------------------------
DATASET_PATH = r'C:\Users\souti\Downloads\archive\Indian-Traffic Sign-Dataset\Images'
CSV_PATH     = r'C:\Users\souti\Downloads\archive\Indian-Traffic Sign-Dataset\traffic_sign.csv'
SAVE_PATH    = 'traffic_classifier.keras'

# -- Hyperparameters ----------------------------------------------------------
IMG_SIZE   = (32, 32)   # 32x32 — better feature retention than 30x30
BATCH_SIZE = 64
EPOCHS     = 20

# -- Load class-name map from CSV ---------------------------------------------
class_names_map = {}
with open(CSV_PATH, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        class_names_map[int(row['ClassId'])] = row['Name']

print("Loading dataset...")

train_ds = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)

val_ds = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)

num_classes = len(train_ds.class_names)
print(f"Found {num_classes} classes.")

# -- tf.data pipeline ---------------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(2000).prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# -- Model --------------------------------------------------------------------
model = tf.keras.Sequential([
    # Normalisation inside the model (consistent with project convention)
    tf.keras.layers.Rescaling(1./255, input_shape=(*IMG_SIZE, 3)),

    # Data augmentation (training-only, ignored at inference)
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomTranslation(0.10, 0.10),

    # Block 1 -- 32 filters
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.25),

    # Block 2 -- 64 filters
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.25),

    # Block 3 -- 128 filters
    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.25),

    # Classifier head
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax'),
], name='indian_tsr_cnn')

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

model.summary()

# -- Callbacks ----------------------------------------------------------------
callbacks = [
    ModelCheckpoint(SAVE_PATH, save_best_only=True, monitor='val_accuracy', verbose=1),
    EarlyStopping(patience=5, monitor='val_accuracy', restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=3, monitor='val_loss', verbose=1),
]

# -- Train --------------------------------------------------------------------
print("\nStarting Training...")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

best_val_acc = max(history.history['val_accuracy'])
print(f"\nBest val accuracy: {best_val_acc:.4f}")
print(f"Model saved to '{SAVE_PATH}'")
