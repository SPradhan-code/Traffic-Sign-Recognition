"""Training entry point:  python -m tsr.train [--config path] [--epochs N]"""
import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from tsr.config import DEFAULT_CONFIG, load_config
from tsr.data import load_training_data, split_data
from tsr.model import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["epochs"] = args.epochs
    keras.utils.set_random_seed(cfg["seed"])

    X, y = load_training_data(cfg["data_dir"], cfg["img_size"], cfg["num_classes"])
    X_tr, X_val, y_tr, y_val = split_data(X, y, cfg["val_split"], cfg["seed"])

    weights = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    class_weight = dict(enumerate(weights))

    model = build_model(cfg["img_size"], cfg["num_classes"])
    model.compile(
        optimizer=keras.optimizers.Adam(cfg["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    Path(cfg["model_path"]).parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            cfg["model_path"], save_best_only=True, monitor="val_accuracy"
        ),
        keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
    ]
    model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        class_weight=class_weight,
        callbacks=callbacks,
    )
    logging.info("Best model saved to %s", cfg["model_path"])


if __name__ == "__main__":
    main()
