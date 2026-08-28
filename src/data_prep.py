"""
data_prep.py
------------
Loads the casting-defect classification dataset and builds PyTorch
DataLoaders for train/val/test.

Real data (recommended): the Kaggle "Casting Product Image Data for Quality
Inspection" dataset (submersible pump impeller photos, labelled ok_front /
def_front). No Kaggle API key needed -- just download the zip from the
Kaggle dataset page in your browser and unzip it under data/casting/.
See README.md "Get the real dataset" for the exact steps and link.

Synthetic fallback: if no real data is found under data/casting/, this
module procedurally generates a small synthetic "ok" vs "defective" image
set (same folder structure) so every script, test, and the Streamlit demo
run end-to-end with zero setup. Swap in the real dataset any time -- no
code changes needed, just add the files and re-run.

Usage:
    from src.data_prep import build_dataloaders
    train_dl, val_dl, test_dl, class_names, using_real = build_dataloaders()
"""

import os
import random

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
REAL_DATA_DIR = os.path.join(DATA_ROOT, "casting")
SYNTHETIC_DATA_DIR = os.path.join(DATA_ROOT, "casting_synthetic")
# NOTE: torchvision's ImageFolder always assigns class indices in *sorted*
# folder-name order, regardless of the order you list them here -- so this
# must stay alphabetical ("def_front" < "ok_front") to match model output
# indices. Getting this wrong silently flips every predicted label. The
# build_dataloaders() assertion below guards against this drifting again.
CLASS_NAMES = ["def_front", "ok_front"]  # index 0 = defective, index 1 = OK
IMG_SIZE = 224
SEED = 42


def _find_real_data_dir(root: str = REAL_DATA_DIR) -> str | None:
    """Search recursively for a directory that directly contains both
    ok_front/ and def_front/ subfolders (the Kaggle dataset can end up
    nested a couple of levels deep depending on how the zip was extracted).
    """
    if not os.path.isdir(root):
        return None
    for dirpath, dirnames, _ in os.walk(root):
        if "ok_front" in dirnames and "def_front" in dirnames:
            ok_count = len(os.listdir(os.path.join(dirpath, "ok_front")))
            def_count = len(os.listdir(os.path.join(dirpath, "def_front")))
            if ok_count > 0 and def_count > 0:
                return dirpath
    return None


def _draw_ok_impeller(draw: ImageDraw.ImageDraw, size: int, rng: random.Random) -> None:
    """A clean top-down 'impeller' shape: a ring of evenly-spaced blades."""
    cx, cy = size // 2, size // 2
    outer_r = int(size * 0.42)
    hub_r = int(size * 0.12)
    base_gray = rng.randint(150, 175)
    draw.ellipse([cx - outer_r, cy - outer_r, cx + outer_r, cy + outer_r], fill=base_gray)
    n_blades = 8
    for i in range(n_blades):
        angle = 2 * np.pi * i / n_blades
        bx = cx + outer_r * 0.75 * np.cos(angle)
        by = cy + outer_r * 0.75 * np.sin(angle)
        blade_gray = base_gray - rng.randint(15, 30)
        draw.ellipse([bx - 10, by - 22, bx + 10, by + 22], fill=blade_gray)
    draw.ellipse([cx - hub_r, cy - hub_r, cx + hub_r, cy + hub_r], fill=base_gray - 45)


def _draw_defect(draw: ImageDraw.ImageDraw, size: int, rng: random.Random) -> None:
    """Overlay a random blow-hole / crack / burr style defect blob."""
    defect_kind = rng.choice(["blowhole", "crack", "burr"])
    cx, cy = size // 2, size // 2
    if defect_kind == "blowhole":
        for _ in range(rng.randint(1, 3)):
            bx = cx + rng.randint(-70, 70)
            by = cy + rng.randint(-70, 70)
            r = rng.randint(4, 12)
            draw.ellipse([bx - r, by - r, bx + r, by + r], fill=rng.randint(20, 45))
    elif defect_kind == "crack":
        x, y = cx + rng.randint(-50, 50), cy + rng.randint(-50, 50)
        for _ in range(rng.randint(6, 12)):
            nx, ny = x + rng.randint(-15, 15), y + rng.randint(-15, 15)
            draw.line([x, y, nx, ny], fill=rng.randint(15, 40), width=2)
            x, y = nx, ny
    else:  # burr
        bx = cx + rng.randint(-80, 80)
        by = cy + rng.randint(-80, 80)
        pts = [(bx + rng.randint(-14, 14), by + rng.randint(-14, 14)) for _ in range(6)]
        draw.polygon(pts, fill=rng.randint(190, 230))


def _generate_synthetic_image(is_defective: bool, rng: random.Random) -> Image.Image:
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=rng.randint(60, 80))
    draw = ImageDraw.Draw(img)
    _draw_ok_impeller(draw, IMG_SIZE, rng)
    if is_defective:
        _draw_defect(draw, IMG_SIZE, rng)
    # Sensor/lighting noise so the task isn't trivially separable by a single pixel rule.
    arr = np.array(img).astype(np.float32)
    arr += rng.gauss(0, 6) + np.random.normal(0, 5, arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L").convert("RGB")
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.9)))
    return img


def generate_synthetic_dataset(n_per_class: int = 220, out_dir: str = SYNTHETIC_DATA_DIR, seed: int = SEED) -> str:
    """Creates data/casting_synthetic/{ok_front,def_front}/*.png if not already present."""
    rng = random.Random(seed)
    np.random.seed(seed)
    for cls, is_defective in [("ok_front", False), ("def_front", True)]:
        cls_dir = os.path.join(out_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        existing = len([f for f in os.listdir(cls_dir) if f.endswith(".png")])
        for i in range(existing, n_per_class):
            img = _generate_synthetic_image(is_defective, rng)
            img.save(os.path.join(cls_dir, f"{cls}_{i:04d}.png"))
    return out_dir


def get_dataset_dir() -> tuple[str, bool]:
    """Returns (dataset_dir, using_real_data)."""
    real_dir = _find_real_data_dir()
    if real_dir:
        return real_dir, True
    synthetic_dir = generate_synthetic_dataset()
    return synthetic_dir, False


def get_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_dataloaders(
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = SEED,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str], bool]:
    """Builds train/val/test DataLoaders from an ImageFolder-structured dataset dir."""
    dataset_dir, using_real = get_dataset_dir()

    # Single ImageFolder load with eval-style transforms for splitting, then a
    # second train-transformed copy so augmentation only applies to the train split.
    base_ds = datasets.ImageFolder(dataset_dir, transform=get_transforms(train=False))
    train_ds_full = datasets.ImageFolder(dataset_dir, transform=get_transforms(train=True))
    # Strict order check (not just set equality) -- ImageFolder assigns model
    # output indices by sorted folder-name order, so CLASS_NAMES must match
    # that exactly or every downstream prediction label silently flips.
    assert base_ds.classes == CLASS_NAMES, (
        f"Expected classes {CLASS_NAMES} (this exact order), found {base_ds.classes} in {dataset_dir}. "
        "Update CLASS_NAMES in data_prep.py to match ImageFolder's sorted order."
    )

    n = len(base_ds)
    n_val = int(n * val_split)
    n_test = int(n * test_split)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx, test_idx = random_split(range(n), [n_train, n_val, n_test], generator=generator)

    train_subset = torch.utils.data.Subset(train_ds_full, list(train_idx))
    val_subset = torch.utils.data.Subset(base_ds, list(val_idx))
    test_subset = torch.utils.data.Subset(base_ds, list(test_idx))

    train_dl = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl, test_dl, base_ds.classes, using_real


if __name__ == "__main__":
    train_dl, val_dl, test_dl, classes, using_real = build_dataloaders()
    print(f"Using {'REAL Kaggle' if using_real else 'SYNTHETIC fallback'} dataset")
    print(f"Classes: {classes}")
    print(f"Train batches: {len(train_dl)}  Val batches: {len(val_dl)}  Test batches: {len(test_dl)}")
