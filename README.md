---
title: Explainable Manufacturing Defect Classifier
emoji: 🔍
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: "1.62.0"
python_version: "3.11"
app_file: src/app.py
pinned: false
---

# Explainable Manufacturing Defect Classifier

**Phase 2 project** of a 6-month self-directed AI/ML continuing-education program (following the UT Austin PGP-AI certificate). This phase levels up the computer-vision skills from the coursework: transfer learning with two backbones (VGG-16 baseline vs. a fine-tuned ResNet50), tracked and compared in MLflow, plus Grad-CAM explainability so a "defective" call comes with a heatmap showing *why* — not just a black-box label. Wrapped as a Flask API and a Streamlit upload-and-explain demo.

> Live demo: (https://huggingface.co/spaces/vinnyb23/explainable-vision-classifier)

![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/pytorch-transfer%20learning-EE4C2C)
![MLflow](https://img.shields.io/badge/tracked%20with-MLflow-0194E2)
![Grad-CAM](https://img.shields.io/badge/explainability-Grad--CAM-9146FF)

## What it does

- **Binary defect classification** — given a photo of a manufactured part, predicts `ok_front` (no defect) vs. `def_front` (defective), the same labeling convention as the popular Kaggle casting-quality-inspection dataset.
- **Two backbones, compared head-to-head** — a VGG-16 baseline (frozen conv base, only the classifier head retrained) vs. a fine-tuned ResNet50 (also unfreezes the last residual block), both logged to MLflow with the same metrics so the "leveled up" claim is measurable, not just asserted.
- **Grad-CAM explainability** — every prediction comes with a heatmap overlay showing which pixels the model actually looked at, using [`pytorch-grad-cam`](https://github.com/jacobgil/pytorch-grad-cam).
- **Flask `/predict` API** — wraps the model + Grad-CAM as a JSON service.
- **Streamlit demo** — upload a part photo, see the prediction, confidence, and heatmap side by side.
- **Zero-setup by default** — no dataset download or API key required to run every script and test. See "Dataset" below.

## Dataset

Real manufacturing-defect image datasets that need a classification label (not segmentation masks) mostly live behind the Kaggle API, which means another credential-setup step. To avoid that friction blocking the rest of the phase, this repo works two ways:

1. **Synthetic fallback (default, zero setup)** — `src/data_prep.py` procedurally generates a small dataset of "impeller"-style parts: a clean ring-of-blades shape for `ok_front`, and the same shape with a randomly placed blow-hole, crack, or burr defect plus sensor noise for `def_front`. It's saved to `data/casting_synthetic/` the first time any script runs, in the exact same folder layout the real dataset uses — so nothing else changes when you swap it in.
2. **Get the real dataset (recommended once you're ready)** — download ["Casting Product Image Data for Quality Inspection"](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) from Kaggle (free account, no API key needed if you use the "Download" button on the web page instead of the API), unzip it, and place the `ok_front/` and `def_front/` folders under `data/casting/` (nested subfolders like `data/casting/casting_data/train/ok_front/` are fine — `data_prep.py` searches for them). `src/data_prep.py` prefers this real data automatically whenever it's present; delete/rename the folder to fall back to synthetic again.

```
data/
├── casting/                    <- put the real Kaggle dataset here (optional, gitignored)
│   ├── ok_front/*.jpg
│   └── def_front/*.jpg
└── casting_synthetic/          <- auto-generated fallback (gitignored, regenerates on demand)
    ├── ok_front/*.png
    └── def_front/*.png
```

## Architecture

```
┌───────────────────┐      ┌────────────────────┐      ┌─────────────────────┐
│  data_prep.py       │ --> │  train.py            │ --> │  models/*.pt          │
│  real data OR        │     │  VGG-16 / ResNet50   │     │  MLflow run (params,  │
│  synthetic fallback │     │  transfer learning    │     │  metrics, artifacts)  │
└───────────────────┘      └────────────────────┘      └──────────┬───────────┘
                                                                     │
                                                                     v
                                                          ┌─────────────────────┐
                                                          │  gradcam.py           │
                                                          │  prediction + heatmap │
                                                          └──────────┬───────────┘
                                                    ┌───────────────┴────────────────┐
                                                    v                                  v
                                          ┌───────────────────┐              ┌───────────────────┐
                                          │  api.py             │              │  app.py             │
                                          │  Flask /predict     │  <────────  │  Streamlit demo     │
                                          └───────────────────┘   (MODEL_MODE=api)  └───────────────────┘
```

## Model comparison (MLflow-tracked, synthetic dataset, 8 epochs each)

| Backbone | Trainable layers | Accuracy | Precision (defect) | Recall (defect) | F1 (defect) | Train time (CPU) |
|---|---|---|---|---|---|---|
| VGG-16 (baseline) | classifier head only | 89.4% | 100.0% | 73.1% | 84.4% | 636 s |
| ResNet50 (fine-tuned) | classifier head + last residual block | 100.0% | 100.0% | 100.0% | 100.0% | 327 s |

_Precision/recall/F1 are scored against the `def_front` (defective) class specifically — the metric that matters for a QA use case is "how good is this model at catching defective parts," not accuracy alone. Regenerate this table any time with `python -m src.train --compare`, which writes `models/model_comparison.csv`._

**Takeaway:** partially unfreezing a deeper backbone (ResNet50's last residual block) gives the model more capacity to adapt to this specific defect-detection task than only retraining VGG-16's final linear layer — it converges to a better score (and trains faster) than the VGG-16 baseline within the same epoch budget, the classic transfer-learning tradeoff between "safe baseline" and "targeted fine-tuning." Take the 100% ResNet50 accuracy with a grain of salt, though — the synthetic dataset's defects (geometric blobs on a clean procedural shape) are easier to separate than real casting photos would be with lighting variation and surface texture; the real next step is re-running this same comparison on the actual Kaggle dataset (see "Dataset" above) to see how much that number drops. See `LEARNING_LOG.md` for the week-by-week writeup.

## Explainability example

Grad-CAM highlights the exact region driving each prediction — here, the heatmap concentrates on the injected crack/blow-hole defect rather than the rest of the part, which is the sanity check that makes a prediction trustworthy instead of a black box.

## Project structure

```
explainable-vision-classifier/
├── README.md
├── LEARNING_LOG.md
├── requirements.txt
├── Dockerfile
├── .env.example
├── .gitignore
├── data/                    <- generated at runtime (gitignored): synthetic images, or drop the real dataset here
├── models/                  <- generated at runtime (gitignored): trained weights, comparison table, sample heatmaps
├── notebooks/
│   └── 01_exploration.ipynb
├── src/
│   ├── data_prep.py         <- real-dataset loader + synthetic fallback generator + DataLoaders
│   ├── train.py             <- VGG-16 / ResNet50 transfer learning + MLflow logging
│   ├── gradcam.py           <- Grad-CAM heatmap generation
│   ├── api.py                <- Flask /predict API
│   └── app.py                <- Streamlit demo
└── tests/
    ├── test_data_prep.py
    ├── test_train.py
    ├── test_gradcam.py
    └── test_api.py
```

## Getting started

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/explainable-vision-classifier.git
cd explainable-vision-classifier
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

CPU-only PyTorch is fine for this project's dataset size. If you have a CUDA GPU, install the matching build from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) instead of the pinned CPU wheel in `requirements.txt`.

### 2. (Optional) get the real dataset

See "Dataset" above. Skip this step to use the synthetic fallback — everything below works either way.

### 3. Train a model

```bash
python -m src.train --backbone resnet50 --epochs 8      # single backbone
python -m src.train --compare --epochs 8                 # both backbones + comparison table
```

Weights are saved to `models/<backbone>.pt`; every run is also logged to MLflow.

### 4. Inspect experiment tracking

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

### 5. Try Grad-CAM on a sample image

```bash
python -m src.gradcam resnet50
```

### 6. Launch the API

```bash
python -m src.api
curl -F "image=@data/casting_synthetic/def_front/def_front_0000.png" http://localhost:5000/predict
```

### 7. Launch the Streamlit demo

```bash
streamlit run src/app.py
```

Set `MODEL_MODE=api` (and `API_URL`) in `.env` to have the Streamlit app call the Flask API instead of loading the model in-process.

### Run with Docker instead

```bash
docker build -t explainable-vision-classifier .
docker run -p 8501:8501 explainable-vision-classifier
```

The image trains a demo ResNet50 model on the synthetic dataset at build time, so the container works out of the box. Mount your own `data/casting/` in before building to train on the real dataset instead.

### Run the tests

```bash
pytest tests/ -v
```

## Roadmap for this repo (Phase 2, Weeks 7–12)

- [x] Week 7: dataset loader (real + synthetic fallback) + VGG-16 baseline
- [x] Week 8: fine-tuned ResNet50 + MLflow before/after comparison
- [x] Week 9: Grad-CAM explainability
- [x] Week 10: Flask `/predict` API
- [x] Week 11: Streamlit upload + prediction + heatmap front end
- [ ] Week 12: deploy a live demo (Hugging Face Spaces), add a screenshot below, push to GitHub

## Screenshots

_Add a screenshot or GIF of the Streamlit app (upload → prediction → heatmap) here once you run it (Week 12 polish task)._

## Part of a larger program

This repo is Phase 2 of a 6-month self-directed AI/ML program:

1. BI + AI Fusion — [`bi-ai-assistant`](https://github.com/Vinnyb23/bi-ai-assistant)
2. **Computer Vision** — this repo
3. Generative AI & Agents — `ai-bi-analyst-agent`
4. MLOps & Deployment (capstone) — `bi-copilot-capstone`, which unifies all three
