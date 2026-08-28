# Learning Log — Phase 2: Explainable Vision Classifier

Running notes as I work through Weeks 7-12 of the program. Format follows the
same pattern as Phase 1's log: what I built, what I learned, what tripped me up.

## Week 7 — Dataset + VGG-16 baseline

**What I built:** `src/data_prep.py`. I wanted a real manufacturing dataset
(the classic Kaggle "casting product quality inspection" set — pump impeller
photos labelled OK vs. defective), but it requires Kaggle API credentials to
download and I didn't want another credential-setup detour this early in the
phase. So the loader checks `data/casting/` first for a real, manually
downloaded dataset (same `ok_front` / `def_front` folder layout Kaggle ships),
and falls back to a **procedurally generated synthetic dataset** if it's not
there — same folder structure, so nothing else in the pipeline has to change.
The synthetic generator draws a simple "impeller" shape (a ring of blades
around a hub) and, for the defective class, overlays a blow-hole, crack, or
burr defect at a random position, plus some Gaussian noise/blur so it's not
trivially separable by a single pixel rule.

**What I learned:** `torchvision.datasets.ImageFolder` + `random_split` is
the simplest way to get train/val/test splits without hand-rolling file
list management, as long as the folder layout is `class_name/*.jpg`.

**Baseline:** VGG-16 pretrained on ImageNet, convolutional base frozen,
only the final classifier layer retrained (see `src/train.py::build_model`).

## Week 8 — Fine-tuned ResNet50 + MLflow comparison

**What I built:** `src/train.py`. Added a second backbone, ResNet50, which
unfreezes the final residual block (`layer4`) as well as the classifier head
— a step up from VGG-16's "just retrain the last layer" baseline. Both
backbones log params/metrics/model artifacts to MLflow under the same
experiment (`explainable-vision-classifier`) so I can compare them side by
side in the MLflow UI, not just from printed numbers.

**What tripped me up:** `mlflow.pytorch.log_model` defaults to a `pt2`
(traced-graph) serialization format in newer MLflow versions, which requires
an `input_example`. I didn't want to wire that up for two backbones with
different input shapes, so I pinned `serialization_format="pickle"` instead
— same approach MLflow used by default in older versions.

**Comparison (synthetic dataset, 8 epochs each, see README for the full
table):** ResNet50 reached 100% test accuracy / F1 in about half the wall
clock time VGG-16 took to reach 89%, which matches the general intuition
that partially unfreezing a deeper, more modern backbone gives it more
capacity to adapt to the new task than only retraining a linear head. I'm
treating ResNet50's 100% with some skepticism though — the synthetic
defects are geometrically simple, so this is more "the easy version of the
task is basically solved" than "this would work in a real QA line." The
real test is re-running the same comparison on the actual Kaggle dataset.

**Bug I caught before shipping (worth remembering):** I originally hard-coded
`CLASS_NAMES = ["ok_front", "def_front"]` in `data_prep.py`, but
`torchvision.datasets.ImageFolder` assigns model output indices by *sorted*
folder name, not the order you list them — so the real index order is
`["def_front", "ok_front"]` ("def" < "ok" alphabetically). Training itself
was unaffected (loss/accuracy used the DataLoader's own label indices
consistently), but every *displayed* prediction in the Grad-CAM/API/Streamlit
layer was silently flipped — a defective part would show up labeled "OK" and
vice versa. Caught it by testing the Grad-CAM output against known-defective
sample images and noticing the label was always wrong. Fixed by reordering
`CLASS_NAMES` to match ImageFolder's real order and adding a strict assertion
in `build_dataloaders()` that fails loudly if it ever drifts again. Lesson:
when a library infers a mapping from string sorting rather than input order,
don't assume your hand-written constant agrees with it — verify against the
library's actual `.classes` attribute, and test the human-readable output,
not just the raw numeric metrics.

## Week 9 — Grad-CAM explainability

**What I built:** `src/gradcam.py`, using the `pytorch-grad-cam` library.
Grad-CAM works by looking at the gradient of the predicted class score with
respect to the activations of a chosen convolutional layer — for VGG-16 that's
the last layer of `model.features`, for ResNet50 it's the last block of
`layer4`. The result is a coarse heatmap over the image showing which regions
most influenced the prediction, which I overlay on the original image so it's
visually obvious.

**Why this matters for the "explainable" part of the project title:** a
defect/no-defect call by itself isn't something a QA engineer should trust
blindly. Being able to show *where* the model is looking is the difference
between "trust me" and an actually auditable decision — this is the whole
point of Week 9 vs. just shipping the Week 8 classifier as-is.

## Week 10 — Flask `/predict` API

**What I built:** `src/api.py`. A `/predict` endpoint accepts a multipart
image upload and returns the predicted class, confidence, and the Grad-CAM
heatmap (base64-encoded PNG) in one JSON response — same pattern as wrapping
any ML model as a service, just returning an extra visual artifact alongside
the prediction.

## Week 11 — Streamlit front end

**What I built:** `src/app.py`. Upload an image, see it side by side with
its Grad-CAM heatmap, and get a plain-language OK / defective verdict with a
confidence score. It can run the model in-process (`MODEL_MODE=local`,
simplest for a demo) or call the Flask API instead (`MODEL_MODE=api`) —
same dual-mode idea Phase 1 didn't need but is useful once the API becomes
its own deployable thing in Week 12.

## Week 12 — Docker, README, deploy

**What I built:** `Dockerfile` (builds the synthetic dataset and trains a
demo ResNet50 model at image-build time so the container works out of the
box), plus this README with the model comparison table and setup
instructions. Pushed to GitHub manually via the web editor (same workflow as
Phase 1's `bi-ai-assistant` repo) and deployed a live demo on Hugging Face
Spaces.

## Swapping in the real Kaggle dataset later

Everything in this repo works unchanged the moment a real dataset lands in
`data/casting/{ok_front,def_front}/*.jpg` — see the README's "Get the real
dataset" section. Worth doing as a follow-up: the real casting photos have
visual detail (surface texture, lighting variation) my synthetic generator
doesn't capture, so accuracy numbers on it will be a better signal of how
this would generalize in an actual QA line.
