"""Convert the Keras model to TensorFlow Lite:  python -m tsr.export_tflite"""
import tensorflow as tf
from tensorflow import keras

from tsr.config import load_config


def main():
    cfg = load_config()
    model = keras.models.load_model(cfg["model_path"], compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # dynamic-range quantization
    data = converter.convert()
    with open(cfg["tflite_path"], "wb") as f:
        f.write(data)
    print(f"Saved {cfg['tflite_path']} ({len(data) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
