import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image

from src.data_prep import CLASS_NAMES
from src.gradcam import explain_image, load_model_for_explain
from src.train import build_model


def test_explain_image_returns_valid_prediction(tmp_path):
    model = build_model("resnet50", num_classes=len(CLASS_NAMES))
    weights_path = tmp_path / "resnet50.pt"
    torch.save(model.state_dict(), weights_path)

    loaded_model = load_model_for_explain("resnet50", str(weights_path))
    fake_image = Image.new("RGB", (300, 300), color=(120, 120, 120))

    pred_class, confidence, overlay_img = explain_image(loaded_model, "resnet50", fake_image)

    assert pred_class in CLASS_NAMES
    assert 0.0 <= confidence <= 1.0
    assert overlay_img.size == (224, 224)
    assert overlay_img.mode == "RGB"
