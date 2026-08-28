import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.train import _evaluate, _run_epoch, build_model


def _fake_loader(n=12, num_classes=2, batch_size=4):
    images = torch.randn(n, 3, 224, 224)
    labels = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


def test_build_model_vgg16_has_correct_output_size():
    model = build_model("vgg16", num_classes=2)
    assert model.classifier[-1].out_features == 2


def test_build_model_resnet50_has_correct_output_size():
    model = build_model("resnet50", num_classes=2)
    assert model.fc.out_features == 2


def test_build_model_unknown_backbone_raises():
    try:
        build_model("not_a_real_backbone")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_run_epoch_and_evaluate_smoke():
    model = build_model("resnet50", num_classes=2)
    criterion = torch.nn.CrossEntropyLoss()
    loader = _fake_loader()

    loss = _run_epoch(model, loader, criterion, optimizer=None)
    assert isinstance(loss, float)
    assert loss >= 0

    metrics = _evaluate(model, loader)
    for key in ("accuracy", "precision", "recall", "f1"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0
