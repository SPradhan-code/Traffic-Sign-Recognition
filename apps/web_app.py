"""Web UI:  streamlit run apps/web_app.py"""
import sys
from pathlib import Path

# Ensure project root and src/ are in sys.path
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
for _p in (str(_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from tsr.config import load_config  # noqa: E402
from tsr.predictor import Predictor  # noqa: E402

CONFIDENCE_THRESHOLD = 0.70

cfg = load_config()


@st.cache_resource
def get_predictor() -> Predictor:
    return Predictor(cfg["model_path"], cfg["img_size"])


def preprocess(image: Image.Image, img_size: int) -> np.ndarray:
    """PIL Image → (1, img_size, img_size, 3) float32 in [0, 255].

    The model contains a Rescaling(1/255) layer, so we pass raw pixel values.
    PIL is natively RGB — no channel swap needed.
    """
    img = image.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    return np.expand_dims(np.array(img, dtype=np.float32), axis=0)


# ── Page layout ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Traffic Sign Recognition", page_icon="🚦")
st.title("🚦 Traffic Sign Recognition")
st.info(
    "Trained on **GTSRB** (43 German traffic-sign classes). "
    "Signs outside this set (e.g. No Honking) will show as **Unrecognized**.",
    icon="ℹ️",
)

# ── Upload & predict ──────────────────────────────────────────────────────────
file = st.file_uploader("Upload a traffic sign image", type=["jpg", "jpeg", "png"])
if file:
    image = Image.open(file)
    st.image(image, width=250)

    predictor = get_predictor()
    tensor = preprocess(image, cfg["img_size"])
    probs = predictor.model.predict(tensor, verbose=0)[0]
    top_conf = float(np.max(probs))

    if top_conf < CONFIDENCE_THRESHOLD:
        st.warning(
            "⚠️ Unrecognized or Out-of-Dataset Traffic Sign "
            f"(best confidence: {top_conf:.1%} < {CONFIDENCE_THRESHOLD:.0%} threshold)."
        )
    else:
        from tsr.labels import CLASS_NAMES

        top_idx = int(np.argmax(probs))
        label = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else str(top_idx)

        st.metric(label="Predicted Class", value=label, delta=f"{top_conf:.1%} confidence")

        st.markdown("**Top-3 predictions**")
        top3 = np.argsort(probs)[::-1][:3]
        for i in top3:
            name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i)
            conf = float(probs[i])
            col1, col2 = st.columns([3, 1])
            col1.progress(conf, text=name)
            col2.write(f"`{conf:.1%}`")
