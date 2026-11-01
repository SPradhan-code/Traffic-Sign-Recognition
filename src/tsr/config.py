"""Configuration loading. All paths in the YAML are relative to the project root."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path=DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("data_dir", "model_path", "tflite_path", "confusion_matrix_path"):
        cfg[key] = str(PROJECT_ROOT / cfg[key])
    return cfg
