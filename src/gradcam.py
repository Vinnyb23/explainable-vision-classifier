"""
gradcam.py
----------
Grad-CAM explainability for the casting-defect classifier (Phase 2, Week 9).

Wraps the `grad-cam` library to produce a heatmap overlay showing which
pixels most influenced the model's prediction -- the whole point of an
"explainable" vision classifier: a defect/no-defect call alone isn't
trustworthy in a QA setting unless you can see *why*.

Usage:
    from src.gradcam import load_model_for_explain, explain_image
    model = load_model_for_explain("resnet50", "models/resnet50.pt")
    pred_class, confidence, overlay_img = explain_image(model, "resnet50", pil_image)
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import models

from src.data_prep import CLASS_NAMES, get_transforms
from src.train import build_model

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def _target_layer(model: torch.nn.Module, backbone: str):
    """The last conv block -- where Grad-CAM gets the richest spatial signal."""
    if backbone == "vgg16":
        return [model.features[-1]]
    if backbone == "resnet50":
        return [model.layer4[-1]]
    raise ValueError(f"Unknown backbone '{backbone}'")


def load_model_for_explain(backbone: str, weights_path: str | None = None) -> torch.nn.Module:
    weights_path = weights_path or os.path.join(MODELS_DIR, f"{backbone}.pt")
    model = build_model(backbone, num_classes=len(CLASS_NAMES))
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def explain_image(model: torch.nn.Module, backbone: str, image: Image.Image) -> tuple[str, float, Image.Image]:
    """Runs prediction + Grad-CAM on a single PIL image.

    Returns (predicted_class_name, confidence, heatmap_overlay_as_PIL_image).
    """
    image_rgb = image.convert("RGB")
    transform = get_transforms(train=False)
    # requires_grad_ is needed even though the backbone weights may be frozen:
    # Grad-CAM needs *an* upstream tensor requiring grad so autograd keeps a
    # graph for the target layer's activations (otherwise, e.g. VGG-16 with a
    # fully-frozen `features` block, the activation gradient comes back as None).
    input_tensor = transform(image_rgb).unsqueeze(0)
    input_tensor.requires_grad_(True)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())

    cam = GradCAM(model=model, target_layers=_target_layer(model, backbone))
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]  # targets=None -> highest-scoring class

    # Overlay needs the *displayed* image resized to match the model's input size, normalized to [0, 1].
    display_img = image_rgb.resize((224, 224))
    rgb_float = np.array(display_img).astype(np.float32) / 255.0
    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    overlay_img = Image.fromarray(overlay)

    return CLASS_NAMES[pred_idx], confidence, overlay_img


if __name__ == "__main__":
    import sys

    from src.data_prep import get_dataset_dir

    backbone = sys.argv[1] if len(sys.argv) > 1 else "resnet50"
    model = load_model_for_explain(backbone)

    dataset_dir, using_real = get_dataset_dir()
    sample_class_dir = os.path.join(dataset_dir, "def_front")
    sample_file = os.listdir(sample_class_dir)[0]
    sample_path = os.path.join(sample_class_dir, sample_file)

    pred, conf, overlay_img = explain_image(model, backbone, Image.open(sample_path))
    out_path = os.path.join(MODELS_DIR, f"gradcam_sample_{backbone}.png")
    overlay_img.save(out_path)
    print(f"Sample: {sample_path}")
    print(f"Prediction: {pred}  (confidence={conf:.3f})")
    print(f"Heatmap saved to {out_path}")
