import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image

import src.api as api
from src.data_prep import CLASS_NAMES
from src.train import build_model


def test_health_endpoint():
    client = api.app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert set(data["classes"]) == set(CLASS_NAMES)


def test_predict_without_image_returns_400():
    client = api.app.test_client()
    resp = client.post("/predict", data={})
    assert resp.status_code == 400


def test_predict_without_trained_model_returns_503(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(api, "_model", None)

    buf = io.BytesIO()
    Image.new("RGB", (50, 50)).save(buf, format="PNG")
    buf.seek(0)

    client = api.app.test_client()
    resp = client.post("/predict", data={"image": (buf, "test.png")}, content_type="multipart/form-data")
    assert resp.status_code == 503


def test_predict_with_trained_model_returns_prediction(tmp_path, monkeypatch):
    model = build_model(api.BACKBONE, num_classes=len(CLASS_NAMES))
    weights_path = tmp_path / f"{api.BACKBONE}.pt"
    torch.save(model.state_dict(), weights_path)

    monkeypatch.setattr(api, "MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(api, "_model", None)

    buf = io.BytesIO()
    Image.new("RGB", (100, 100), color=(80, 80, 80)).save(buf, format="PNG")
    buf.seek(0)

    client = api.app.test_client()
    resp = client.post("/predict", data={"image": (buf, "test.png")}, content_type="multipart/form-data")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["prediction"] in CLASS_NAMES
    assert 0.0 <= data["confidence"] <= 1.0
    assert "heatmap_base64" in data
