# DeepGuard — Video Deepfake Detection

**Updated problem statement:** a user uploads a video; the application examines visual
facial evidence and returns Likely real, Likely fake, Not sure, or Cannot analyze, with
an understandable model score, face samples and a downloadable report.

This is a **video-only student research pilot**, not a universally reliable detector.
The expanded inventory contains 822 unique files. A balanced selection contains
**319 real and 319 fake files**, including 102 originals and 217 augmentations per class.
Another 184 fake files are retained in reserve. Quality filtering and split-level
balancing determine which files enter fitting and evaluation. Current measured
results are in EVALUATION_REPORT.md and the app's reliability page. Accuracy on
arbitrary unfamiliar internet videos has not been established.

## Run the prepared application

```bash
cd /workspace/Deepfake_Detection
bash scripts/run.sh
```

The application listens on port 8501. Open that port through your environment's supported
port-forwarding interface. No localhost preview link is assumed. On a local computer,
run Streamlit from the checkout after following INSTALLATION.md.

## Reproduce

```bash
bash scripts/setup.sh
source /workspace/deepguard-venv/bin/activate
export TORCH_HOME=/workspace/deepguard-runtime/torch
export MPLCONFIGDIR=/workspace/deepguard-runtime/matplotlib
# Once authorized videos and the labeled, grouped manifest exist:
deepguard train data/manifests/splits.csv --balanced
deepguard robustness
bash scripts/validate.sh
```

Raw owner videos and private manifests are local/ignored. The trained standard head and quality-triggered retry head
are included as `checkpoints/visual.json` and `checkpoints/coverage_retry.json`; the official ImageNet backbone downloads through
torchvision on first inference if not cached. The checksummed OpenCV SSD face detector
is bundled in assets/. A fresh clone still needs FFprobe on PATH. The
manifest must retain real/fake pairs and identical content in one split. Filename labels
were supplied by the owner; benchmark origin and subject identities were not verified.

## Design

OpenCV's neural SSD face detector samples faces across the full clip. Official ImageNet
EfficientNet-B0 features are frozen; validation selects between five regularized linear
classifiers and two small neural classifiers using mean/variation features and texture
statistics. This is transfer learning; the image backbone is not fine-tuned. Training-only scaling, separate validation,
calibration and held-out test partitions prevent direct source-pair leakage. Identity/
donor-disjoint evaluation remains unverified. Matching vN/vsN families and their
augmentations stay in one split. Old split assignments are preserved. Primary testing
uses reserved originals from the expanded sources; these examples have now been inspected and provide regression monitoring. Fresh external evaluation is still required.
Temperature scaling is evaluated separately;
the application displays a raw classifier score, never a guaranteed probability of truth.

## Project map

| Location | Role |
|---|---|
| `app/main.py`, `app/styles/` | Six-page, plain-language interface |
| `src/deepguard/` | Video decoding, model, training, evaluation, reports and CLI |
| `configs/default.yaml` | Sampling, size limits and uncertainty thresholds |
| `data/raw/` | Owner uploads and supplied augmentations; ignored |
| `assets/` | Bundled neural face detector and upstream notices |
| `data/manifests/`, `data/processed/` | Grouped splits and reusable features; ignored |
| `checkpoints/visual.json` | Actual fitted visual head, included for inference |
| `reports/results/` | Real held-out predictions, comparisons, exclusions and robustness |
| `tests/` | Unit, interface, failure-case and actual held-out integration checks |
| `scripts/` | Reproducible setup, training, checking and startup |

Read ARCHITECTURE.md, TRAINING_GUIDE.md and REQUIREMENTS_TRACEABILITY.md for implementation
and scope details. Temporary uploaded files are deleted after analysis. Session previews
and extracted images remain until cleared or the session ends. The initial implementation
was AI-assisted; actual human work belongs in TEAM_CONTRIBUTIONS_TEMPLATE.md.
