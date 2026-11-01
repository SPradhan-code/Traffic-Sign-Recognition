"""Evaluate on the official GTSRB test set:  python -m tsr.evaluate"""

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402
from tensorflow import keras  # noqa: E402

from tsr.config import load_config  # noqa: E402
from tsr.data import read_image  # noqa: E402

logging.basicConfig(level=logging.INFO)


def evaluate():
    cfg = load_config()
    model = keras.models.load_model(cfg["model_path"], compile=False)
    df = pd.read_csv(f"{cfg['data_dir']}/Test.csv")

    X, y_true = [], []
    for path, cid in zip(df["Path"], df["ClassId"]):
        img = read_image(f"{cfg['data_dir']}/{path}", cfg["img_size"])
        if img is not None:
            X.append(img)
            y_true.append(cid)
    y_pred = model.predict(np.array(X), batch_size=256).argmax(axis=1)

    acc = float(np.mean(np.array(y_true) == y_pred))
    print(f"\nTest accuracy: {acc:.4f}\n")
    print(classification_report(y_true, y_pred, digits=3))

    plt.figure(figsize=(14, 12))
    sns.heatmap(confusion_matrix(y_true, y_pred), cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("GTSRB confusion matrix")
    plt.savefig(cfg["confusion_matrix_path"], dpi=150, bbox_inches="tight")
    print("Saved", cfg["confusion_matrix_path"])


if __name__ == "__main__":
    evaluate()
