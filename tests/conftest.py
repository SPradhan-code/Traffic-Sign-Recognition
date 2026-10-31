import numpy as np
import pytest

from tsr.model import build_model


@pytest.fixture(scope="session")
def tiny_model_path(tmp_path_factory):
    """An untrained model is enough to test plumbing; no dataset needed."""
    path = tmp_path_factory.mktemp("models") / "tiny.keras"
    build_model(32, 43).save(path)
    return str(path)


@pytest.fixture
def random_image():
    return np.random.randint(0, 255, (50, 60, 3), dtype="uint8")
