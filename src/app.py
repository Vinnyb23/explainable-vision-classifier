"""
app.py
------
Streamlit front end for the explainable casting-defect classifier
(Phase 2, Week 11). Upload a part photo, get a prediction and a Grad-CAM
heatmap showing which pixels drove that call.

Two modes, controlled by the MODEL_MODE env var:
  - "local" (default): loads the model in-process (simplest for a demo/HF Space).
  - "api": calls a running Flask /predict endpoint instead (set API_URL).

Usage:
    streamlit run src/app.py
"""

import base64
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import streamlit as st
from PIL import Image

from src.data_prep import CLASS_NAMES, get_dataset_dir
from src.gradcam import explain_image, load_model_for_explain

MODEL_MODE = os.environ.get("MODEL_MODE", "local")
API_URL = os.environ.get("API_URL", "http://localhost:5000")
BACKBONE = os.environ.get("MODEL_BACKBONE", "resnet50")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

st.set_page_config(page_title="Explainable Defect Classifier", page_icon="🔍", layout="centered")


@st.cache_resource
def _load_model():
    weights_path = os.path.join(MODELS_DIR, f"{BACKBONE}.pt")
    if not os.path.exists(weights_path):
        return None
    return load_model_for_explain(BACKBONE, weights_path)


def predict_local(image: Image.Image):
    model = _load_model()
    if model is None:
        return None
    pred_class, confidence, overlay_img = explain_image(model, BACKBONE, image)
    return pred_class, confidence, overlay_img


def predict_via_api(image: Image.Image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    resp = requests.post(f"{API_URL}/predict", files={"image": ("image.png", buf, "image/png")}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    overlay_bytes = base64.b64decode(data["heatmap_base64"])
    overlay_img = Image.open(io.BytesIO(overlay_bytes))
    return data["prediction"], data["confidence"], overlay_img


st.title("🔍 Explainable Manufacturing Defect Classifier")
st.caption(
    "Phase 2 project: transfer learning (VGG-16 baseline vs. fine-tuned ResNet50) "
    "+ Grad-CAM explainability, wrapped as a Flask API and this Streamlit demo."
)

with st.sidebar:
    st.header("About")
    st.write(f"**Backbone:** `{BACKBONE}`")
    st.write(f"**Mode:** `{MODEL_MODE}`")
    weights_path = os.path.join(MODELS_DIR, f"{BACKBONE}.pt")
    if MODEL_MODE == "local" and not os.path.exists(weights_path):
        st.warning(
            f"No trained weights found at `models/{BACKBONE}.pt`. "
            f"Run `python -m src.train --backbone {BACKBONE}` first."
        )
    st.markdown("---")
    st.write("Classes: `ok_front` (no defect) vs `def_front` (defective).")

uploaded_file = st.file_uploader("Upload a part image", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    with col1:
        st.subheader("Input image")
        st.image(image, use_container_width=True)

    with st.spinner("Running inference + Grad-CAM..."):
        try:
            if MODEL_MODE == "api":
                result = predict_via_api(image)
            else:
                result = predict_local(image)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            result = None

    if result is None:
        st.info("Train a model first (see sidebar), or set MODEL_MODE=api with a running Flask server.")
    else:
        pred_class, confidence, overlay_img = result
        with col2:
            st.subheader("Grad-CAM heatmap")
            st.image(overlay_img, use_container_width=True)

        if pred_class == "def_front":
            st.error(f"**Defective** (confidence: {confidence:.1%})")
        else:
            st.success(f"**OK — no defect detected** (confidence: {confidence:.1%})")
else:
    st.info("Upload an image above, or try a sample from the dataset:")
    dataset_dir, using_real = get_dataset_dir()
    st.caption(f"Currently using {'the real Kaggle dataset' if using_real else 'the synthetic fallback dataset'}.")
    sample_cols = st.columns(4)
    for i, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(dataset_dir, cls)
        if os.path.isdir(cls_dir):
            files = sorted(os.listdir(cls_dir))[:2]
            for j, fname in enumerate(files):
                with sample_cols[(i * 2 + j) % 4]:
                    st.image(os.path.join(cls_dir, fname), caption=cls, use_container_width=True)
