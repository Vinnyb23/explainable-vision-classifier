"""
api.py
------
Flask API wrapping the trained classifier + Grad-CAM explainability
(Phase 2, Week 10). This is the same "wrap the model as a service" step
from Phase 1's BI copilot, applied to computer vision.

Endpoints:
    GET  /health            -> {"status": "ok", "backbone": "resnet50"}
    POST /predict            -> multipart/form-data "image" file
                                 returns {"prediction", "confidence", "heatmap_base64"}

Usage:
    python -m src.api
    curl -F "image=@some_part.jpg" http://localhost:5000/predict
"""

import base64
import io
import os

from flask import Flask, jsonify, request
from PIL import Image

from src.data_prep import CLASS_NAMES
from src.gradcam import explain_image, load_model_for_explain

BACKBONE = os.environ.get("MODEL_BACKBONE", "resnet50")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

app = Flask(__name__)
_model = None


def get_model():
    global _model
    if _model is None:
        weights_path = os.path.join(MODELS_DIR, f"{BACKBONE}.pt")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"No trained weights found at {weights_path}. Run `python -m src.train --backbone {BACKBONE}` first."
            )
        _model = load_model_for_explain(BACKBONE, weights_path)
    return _model


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "backbone": BACKBONE, "classes": CLASS_NAMES})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Send a multipart/form-data request with an 'image' file field."}), 400

    file = request.files["image"]
    try:
        image = Image.open(io.BytesIO(file.read()))
    except Exception as exc:
        return jsonify({"error": f"Could not read image: {exc}"}), 400

    try:
        model = get_model()
        pred_class, confidence, overlay_img = explain_image(model, BACKBONE, image)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

    buf = io.BytesIO()
    overlay_img.save(buf, format="PNG")
    heatmap_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return jsonify({
        "prediction": pred_class,
        "is_defective": pred_class == "def_front",
        "confidence": round(confidence, 4),
        "heatmap_base64": heatmap_base64,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
