"""Data loading and splitting."""
import logging
import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)


def read_image(path: str, img_size: int):
    """Read an image from disk -> RGB uint8 array of shape (img_size, img_size, 3)."""
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (img_size, img_size))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_training_data(data_dir: str, img_size: int, num_classes: int):
    images, labels = [], []
    for class_id in range(num_classes):
        class_dir = os.path.join(data_dir, "Train", str(class_id))
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"{class_dir} not found. Run scripts/download_data.sh or see the README."
            )
        for fname in sorted(os.listdir(class_dir)):
            img = read_image(os.path.join(class_dir, fname), img_size)
            if img is None:
                log.warning("Skipping unreadable file %s", fname)
                continue
            images.append(img)
            labels.append(class_id)
    log.info("Loaded %d training images", len(images))
    return np.array(images, dtype="uint8"), np.array(labels, dtype="int64")


def split_data(X, y, val_split: float, seed: int):
    # stratify keeps class proportions identical in both splits
    return train_test_split(X, y, test_size=val_split, stratify=y, random_state=seed)
