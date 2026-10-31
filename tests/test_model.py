import numpy as np

from tsr.labels import CLASS_NAMES
from tsr.model import build_model


def test_label_count():
    assert len(CLASS_NAMES) == 43


def test_model_output_is_probability_distribution():
    model = build_model(32, 43)
    out = model.predict(np.zeros((2, 32, 32, 3), dtype="float32"), verbose=0)
    assert out.shape == (2, 43)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-4)
