import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_prep import (
    CLASS_NAMES,
    build_dataloaders,
    generate_synthetic_dataset,
    get_dataset_dir,
)


def test_generate_synthetic_dataset_creates_both_classes(tmp_path):
    out_dir = generate_synthetic_dataset(n_per_class=6, out_dir=str(tmp_path / "synth"))
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(out_dir, cls)
        assert os.path.isdir(cls_dir)
        files = [f for f in os.listdir(cls_dir) if f.endswith(".png")]
        assert len(files) == 6


def test_generate_synthetic_dataset_is_idempotent(tmp_path):
    out_dir = str(tmp_path / "synth")
    generate_synthetic_dataset(n_per_class=5, out_dir=out_dir)
    generate_synthetic_dataset(n_per_class=5, out_dir=out_dir)  # should not duplicate/error
    for cls in CLASS_NAMES:
        files = [f for f in os.listdir(os.path.join(out_dir, cls)) if f.endswith(".png")]
        assert len(files) == 5


def test_get_dataset_dir_falls_back_to_synthetic():
    dataset_dir, using_real = get_dataset_dir()
    assert os.path.isdir(dataset_dir)
    for cls in CLASS_NAMES:
        assert os.path.isdir(os.path.join(dataset_dir, cls))
    assert isinstance(using_real, bool)


def test_build_dataloaders_returns_correct_shapes():
    train_dl, val_dl, test_dl, classes, using_real = build_dataloaders(batch_size=8)
    assert set(classes) == set(CLASS_NAMES)
    assert len(train_dl.dataset) > 0
    assert len(val_dl.dataset) > 0
    assert len(test_dl.dataset) > 0

    images, labels = next(iter(train_dl))
    assert images.shape[1:] == (3, 224, 224)
    assert labels.min() >= 0 and labels.max() <= 1
