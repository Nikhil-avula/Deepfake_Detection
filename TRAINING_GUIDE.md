# Reproduce video-only training

Run from the checkout after authorized files and the grouped manifest exist:

```bash
source /workspace/deepguard-venv/bin/activate
export TORCH_HOME=/workspace/deepguard-runtime/torch
export MPLCONFIGDIR=/workspace/deepguard-runtime/matplotlib
deepguard train data/manifests/splits.csv
deepguard robustness
bash scripts/validate.sh
```

The command samples/caches actual face features, fits five regularization candidates
using train rows, selects by validation ROC-AUC (validation Brier score breaks ties),
fits temperature using calibration rows, then measures untouched source-pair test rows.
Two texture baselines use the same train/test split. No full neural-network fine-tuning
or multi-epoch backbone optimization is claimed. The fit is supervised learning of the
visual decision head; ImageNet weights are generic initialization.

Config fixes seed 42, sixteen whole-clip samples, 224-pixel crops and CPU batch size 8.
Features and all preprocessing gates are identical for inference and training. Candidate
C values are .001, .01, .1, 1 and 10. Feature scaling is fitted only on training data.
Raw and separately temperature-scaled results are both retained; the app uses raw scores.

Artifacts: checkpoints/visual.json contains the actual classifier/scaler parameters,
config, training counts, manifest hash and provenance; data/manifests/prepared.csv lists
quality eligibility; reports/results contains model selection, predictions, metrics,
curves, baseline comparison and exclusions. The application automatically sees checkpoint
modification changes and invalidates its cached model. Raw owner data and artifacts are
ignored by Git, so a fresh clone needs those local prerequisites or retraining.

The final held-out split was not used for candidate selection. During development an
initial quality policy rejected an entire video after one ambiguous sampled frame;
preprocessing was corrected to skip such frames and reject videos when over 25% were
ambiguous, before inspecting final held-out results. This is an engineering policy,
not an accuracy-optimized test threshold. Calibration overconfidence is reported rather
than hidden. Further changes should use new independent external evaluation data.
