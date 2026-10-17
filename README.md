# DeepGuard AI

Explainable multimodal deepfake research in Python and Streamlit. The project examines
face artifacts, audio spectra, temporal visual embeddings and learned audio–mouth
alignment, then compares individual evidence with fusion.

**Current state:** implemented research software, not a validated detector. No dataset,
trained detection weights or benchmark results are bundled. FakeAVCeleb access requires
approval; the user confirmed they do not yet have access. The interface honestly shows
“Model not trained”. See [PROJECT_STATUS.md](PROJECT_STATUS.md) for executed checks and
[REQUIREMENTS_TRACEABILITY.md](REQUIREMENTS_TRACEABILITY.md) for remaining work.

## Quick start on this prepared Linux machine

```bash
cd /workspace/Deepfake_Detection
source /workspace/deepguard-venv/bin/activate
pytest -q
streamlit run app/main.py --server.headless true --server.address 127.0.0.1 --server.port 8501 --server.maxUploadSize 100 --browser.gatherUsageStats false
```

See [INSTALLATION.md](INSTALLATION.md) for fresh Linux/Windows installation,
[DATASET_GUIDE.md](DATASET_GUIDE.md) for authorized acquisition and label contracts,
[TRAINING_GUIDE.md](TRAINING_GUIDE.md) for exact experiment commands and
[EVALUATION_REPORT.md](EVALUATION_REPORT.md) for the distinction between engineering
verification and research evidence.

## How it works

1. Obtain permitted data and genuine modality labels; inventory every file.
2. Group all identities, source recordings and identical content before splitting.
3. Decode bounded video, track one face conservatively, extract log-Mel audio and
   landmark-aligned mouth/audio windows when the optional landmark asset exists.
4. Train classical baselines, EfficientNet, audio CNN, visual-embedding LSTM and a
   learned two-stream synchronization model. Evaluate synchronization on aligned and
   deliberately offset genuine pairs, separately from fake classification.
5. Collect modality evidence on a reserved fusion split; learn logistic late fusion
   with quality and availability covariates. Fit temperature on a separate calibration
   split. Compare against averaging and quality-weighted log-odds.
6. Evaluate unseen test groups and external datasets without retuning on them.
7. Upload authorized media, inspect evidence and download an escaped HTML report.
   Without calibrated fusion or adequate evidence, abstain.

## Repository map

| Location | Purpose |
|---|---|
| `src/deepguard/` | Data, media, model, training, fusion, evaluation, attribution, reporting and CLI modules |
| `app/` | Eight-section Streamlit dashboard and theme |
| `configs/default.yaml` | Reproducible processing and optimization settings |
| `scripts/` | Setup and validation helpers |
| `tests/` | Engineering tests; generated fixtures are not dataset benchmarks |
| `data/` | Local authorized media, manifests and caches, ignored where large/private |
| `checkpoints/` | Local trained weights and optional landmark asset, ignored |
| `reports/` | Measured outputs; no invented research results |
| `experiments/` | Environment and execution records |

The proposed contribution is **quality- and availability-aware late fusion**. It is a
testable experimental hypothesis, not a claim of a new state of the art. Synthetic
engineering fixtures must never be reported as deepfake detection performance.

Temporary upload files are deleted when analysis exits, including exceptions. Extracted
evidence remains in Streamlit session memory until cleared/disconnected. Dataset caches
persist intentionally and should only contain licensed research data.
