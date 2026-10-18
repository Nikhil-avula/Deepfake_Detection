# DeepGuard — Video Deepfake Detection

**Updated problem statement:** a user uploads a video; the application examines visual
facial evidence and returns Likely real, Likely fake, Not sure, or Cannot analyze, with
an understandable model score, face samples and a downloadable report.

This is a **video-only student research pilot**, not a universally reliable detector.
The owner provided 49 real-labeled and 49 fake-labeled videos. We trained an actual
classifier and evaluated reserved source groups. The observed test accuracy is **60% on
10 analyzable videos**, with **62.5% test coverage**. It cannot promise correct answers on
arbitrary unfamiliar videos. See EVALUATION_REPORT.md before interpreting the demo.

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
deepguard train data/manifests/splits.csv
deepguard robustness
bash scripts/validate.sh
```

Raw owner videos, private manifests and model artifacts are local/ignored. A fresh Git
clone does not include them or automatically reproduce the owner's trained model. The
manifest must retain real/fake pairs and identical content in one split. Filename labels
were supplied by the owner; benchmark origin and subject identities were not verified.

## Design

OpenCV samples frames across the full clip and crops single faces. Official ImageNet
EfficientNet-B0 features are frozen; a regularized visual classifier is fitted on mean/
variation features and texture statistics. This is transfer learning, not a claim that
we fine-tuned the entire neural backbone. Training-only scaling, separate validation,
calibration and held-out test partitions prevent direct source-pair leakage. Identity/
donor-disjoint evaluation remains unverified. Temperature scaling is evaluated separately;
the application displays a raw classifier score, never a guaranteed probability of truth.

## Project map

| Location | Role |
|---|---|
| `app/main.py`, `app/styles/` | Six-page, plain-language interface |
| `src/deepguard/` | Video decoding, model, training, evaluation, reports and CLI |
| `configs/default.yaml` | Sampling, size limits and uncertainty thresholds |
| `data/raw/uploaded/` | Owner's 98 video files; ignored |
| `data/manifests/`, `data/processed/` | Grouped splits and reusable features; ignored |
| `checkpoints/visual.json` | Actual fitted visual head; ignored |
| `reports/results/` | Real held-out predictions, comparisons, exclusions and robustness |
| `tests/` | Unit, interface, failure-case and actual held-out integration checks |
| `scripts/` | Reproducible setup, training, checking and startup |

Read ARCHITECTURE.md, TRAINING_GUIDE.md and REQUIREMENTS_TRACEABILITY.md for implementation
and scope details. Temporary uploaded files are deleted after analysis. Session previews
and extracted images remain until cleared or the session ends. The initial implementation
was AI-assisted; actual human work belongs in TEAM_CONTRIBUTIONS_TEMPLATE.md.
