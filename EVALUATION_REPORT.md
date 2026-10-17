# Evaluation report

**Detection metrics: unavailable.** No authorized dataset, trained detector or held-out
research predictions exist. This is not zero accuracy, and no example metrics are used
as measured results. Generated color/sine engineering fixtures are explicitly excluded
from research reporting.

The initial engineering suite ran `pytest -q`: **23 passed in 23.88 seconds**, with 18
upstream warnings. It checked connected identity/source grouping, duplicate leakage
logic, modality-label separation, metric known answers, model tensor shapes/gradients,
audio timestamp windows, synchronization shift/gap representation, genuine FFmpeg
decoding of generated fixtures, no-audio/corrupt/no-face cases, cache repeatability,
abstention without weights, HTML escaping and all eight Streamlit pages. Subsequent
expanded runs and exact outcomes are recorded in PROJECT_STATUS.

## Planned real evaluation

Run TRAINING_GUIDE commands only with audited data. Each prediction CSV records file,
label, split, logit, score and forward time. Sidecars record checkpoint and manifest
hashes, model size and coverage. `evaluate` accepts only explicit held-out test rows and
refuses to compare raw sync mismatch to overall fake labels. Metrics JSON and curves
come from those predictions. Baseline results similarly retain file-level predictions.

Accuracy is fraction correct; precision is correctness among predicted fakes; recall is
fake detection rate; F1 balances precision/recall. ROC-AUC measures ranking across
thresholds; PR-AUC integrates the precision–recall curve and differs from average
precision (both are named separately). Confusion matrices use rows=true and columns=
predicted with 0=real,1=fake. FPR is false fake alarms among real media; FNR is missed
fakes among fake media. Balanced accuracy averages class recalls. Brier score measures
squared probability error; ECE compares observed frequency and scores in ten equal-width
bins. Undefined single-class ROC/PR AUC is null, never silently reported as perfect.

Group-bootstrap accuracy intervals are available for at least three groups. Generated
split manifests retain connected-component IDs propagated to prediction tables for
grouped resampling. Custom manifests lacking these IDs fall back to exact identity-set
strings; do not describe those as independent subject intervals when sources overlap.
All confidence intervals depend on sampling assumptions and dataset coverage.

Runtime currently distinguishes forward time in individual predictions from end-to-end
analysis time. Peak CPU RSS/GPU VRAM profiling and more extensive uncertainty intervals
remain pending. Model bytes are recorded, not inferred from an architecture name.

Compare only identical held-out targets with compatible labels. Report exclusions,
coverage and abstention separately. Do not collapse skipped/missing evidence into correct
predictions. Cross-dataset and unseen-method studies, calibration before/after comparisons,
sync offset controls, retrained ablations and robustness curves remain unrun.
