"""Web UI:  streamlit run apps/web_app.py"""
import numpy as np
import streamlit as st
from PIL import Image

from tsr.config import load_config
from tsr.predictor import Predictor

cfg = load_config()


@st.cache_resource
def get_predictor():
    return Predictor(cfg["model_path"], cfg["img_size"])


st.set_page_config(page_title="Traffic Sign Recognition", page_icon="🚦")
st.title("🚦 Traffic Sign Recognition")
st.caption("CNN trained on GTSRB (43 classes)")

file = st.file_uploader("Upload a traffic sign image", type=["jpg", "jpeg", "png"])
if file:
    image = Image.open(file).convert("RGB")
    st.image(image, width=250)
    for p in get_predictor().predict(np.array(image)):
        st.write(f"**{p['label']}**: {p['confidence']:.1%}")
        st.progress(min(max(p["confidence"], 0.0), 1.0))
