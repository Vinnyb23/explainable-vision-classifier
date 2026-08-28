"""
train.py
--------
Transfer-learning training for the casting-defect binary classifier.

Trains two backbones on the same data/split so they can be compared
head-to-head (Phase 2, Week 7-8 checkpoint):
  - vgg16    -- the baseline approach from the UT Austin coursework
  - resnet50 -- the "leveled up" modern backbone

Both freeze the pretrained convolutional base and fine-tune a small
classifier head, then log params/metrics/model artifacts to MLflow so the
before/after comparison has a durable record (not just printed numbers).

Usage:
    python -m src.train --backbone vgg16
    python -m src.train --backbone resnet50
    python -m src.train --compare        # trains both, writes comparison table
"""

import argparse
import os
import time
import warnings

import mlflow
import mlflow.pytorch
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torchvision import models

from src.data_prep import CLASS_NAMES, build_dataloaders

warnings.filterwarnings("ignore")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MLRUNS_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "mlruns.db")
COMPARISON_PATH = os.path.join(MODELS_DIR, "model_comparison.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BACKBONES = ["vgg16", "resnet50"]


def build_model(backbone: str, num_classes: int = 2) -> nn.Module:
    if backbone == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        for param in model.features.parameters():
            param.requires_grad = False
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for name, param in model.named_parameters():
            if not name.startswith("fc") and not name.startswith("layer4"):
                param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone '{backbone}'. Choose from {BACKBONES}.")
    return model.to(DEVICE)


def _run_epoch(model, loader, criterion, optimizer=None) -> float:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if is_train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


# ImageFolder assigns indices by sorted folder name, so "def_front" (the class
# we actually care about catching) is always index 0 -- see the CLASS_NAMES
# note in data_prep.py. Scoring precision/recall/f1 against that class (rather
# than sklearn's default pos_label=1) reports "how good is this model at
# catching defective parts", which is the metric that matters for QA.
DEFECT_LABEL = CLASS_NAMES.index("def_front")


@torch.no_grad()
def _evaluate(model, loader) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, pos_label=DEFECT_LABEL, zero_division=0),
        "recall": recall_score(all_labels, all_preds, pos_label=DEFECT_LABEL, zero_division=0),
        "f1": f1_score(all_labels, all_preds, pos_label=DEFECT_LABEL, zero_division=0),
    }


def train_and_log(backbone: str, epochs: int = 5, lr: float = 1e-3, batch_size: int = 32) -> dict:
    os.makedirs(MODELS_DIR, exist_ok=True)
    train_dl, val_dl, test_dl, classes, using_real = build_dataloaders(batch_size=batch_size)

    model = build_model(backbone, num_classes=len(classes))
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    mlflow.set_tracking_uri(f"sqlite:///{MLRUNS_DB_PATH}")
    mlflow.set_experiment("explainable-vision-classifier")

    with mlflow.start_run(run_name=f"{backbone}_transfer_learning"):
        mlflow.log_params({
            "backbone": backbone,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "using_real_data": using_real,
            "n_train": len(train_dl.dataset),
            "n_val": len(val_dl.dataset),
            "n_test": len(test_dl.dataset),
        })

        start = time.time()
        for epoch in range(1, epochs + 1):
            train_loss = _run_epoch(model, train_dl, criterion, optimizer)
            val_loss = _run_epoch(model, val_dl, criterion, optimizer=None)
            val_metrics = _evaluate(model, val_dl)
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_metrics["accuracy"]},
                step=epoch,
            )
            print(
                f"[{backbone}] epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_metrics['accuracy']:.3f}"
            )
        train_time = time.time() - start

        test_metrics = _evaluate(model, test_dl)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metric("train_time_sec", train_time)
        mlflow.pytorch.log_model(model, name="model", serialization_format="pickle")

    weights_path = os.path.join(MODELS_DIR, f"{backbone}.pt")
    torch.save(model.state_dict(), weights_path)

    result = {
        "backbone": backbone,
        "using_real_data": using_real,
        "train_time_sec": round(train_time, 1),
        **{k: round(v, 4) for k, v in test_metrics.items()},
    }
    print(f"Saved weights to {weights_path}")
    return result


def compare_backbones(epochs: int = 5) -> pd.DataFrame:
    rows = [train_and_log(backbone, epochs=epochs) for backbone in BACKBONES]
    df = pd.DataFrame(rows)
    df.to_csv(COMPARISON_PATH, index=False)
    print(f"\nModel comparison saved to {COMPARISON_PATH}\n")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the casting-defect classifier.")
    parser.add_argument("--backbone", choices=BACKBONES, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--compare", action="store_true", help="Train all backbones and write a comparison table.")
    args = parser.parse_args()

    if args.compare or args.backbone is None:
        compare_backbones(epochs=args.epochs)
    else:
        train_and_log(args.backbone, epochs=args.epochs)
