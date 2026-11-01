"""CNN architecture. Rescaling + augmentation live inside the model so that
serving can never drift from training preprocessing."""

from tensorflow import keras
from tensorflow.keras import layers


def build_model(img_size: int = 32, num_classes: int = 43) -> keras.Model:
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)

    # Augmentation is only active during training.
    # No horizontal flip: it would turn "turn left" into "turn right".
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomTranslation(0.1, 0.1)(x)

    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="tsr_cnn")
