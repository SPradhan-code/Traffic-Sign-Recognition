from tsr.predictor import Predictor


def test_predict_returns_sorted_topk(tiny_model_path, random_image):
    result = Predictor(tiny_model_path, 32).predict(random_image, top_k=3)
    assert len(result) == 3
    confs = [r["confidence"] for r in result]
    assert confs == sorted(confs, reverse=True)
    assert all(0.0 <= c <= 1.0 for c in confs)
    assert all({"class_id", "label", "confidence"} <= set(r) for r in result)
