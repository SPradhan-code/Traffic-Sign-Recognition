import cv2
from fastapi.testclient import TestClient


def _client(tiny_model_path, monkeypatch):
    monkeypatch.setenv("TSR_MODEL_PATH", tiny_model_path)
    from apps import api

    api.get_predictor.cache_clear()
    return TestClient(api.app)


def test_health(tiny_model_path, monkeypatch):
    assert _client(tiny_model_path, monkeypatch).get("/health").json() == {"status": "ok"}


def test_predict_ok(tiny_model_path, monkeypatch, random_image):
    ok, buf = cv2.imencode(".png", random_image)
    r = _client(tiny_model_path, monkeypatch).post(
        "/predict", files={"file": ("a.png", buf.tobytes(), "image/png")}
    )
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 3


def test_predict_rejects_bad_type(tiny_model_path, monkeypatch):
    r = _client(tiny_model_path, monkeypatch).post(
        "/predict", files={"file": ("a.txt", b"hello", "text/plain")}
    )
    assert r.status_code == 400


def test_predict_rejects_corrupt_image(tiny_model_path, monkeypatch):
    r = _client(tiny_model_path, monkeypatch).post(
        "/predict", files={"file": ("a.png", b"not an image", "image/png")}
    )
    assert r.status_code == 400
