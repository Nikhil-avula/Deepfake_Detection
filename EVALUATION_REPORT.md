# V004 — Two-pass coverage and performance evaluation

## Behavior

The existing standard classifier and its exact 16-frame preprocessing are preserved.
Only a Cannot analyze result triggers a second search: 24 whole-clip samples, sequential
decoding, at least four usable face crops and 15% coverage. A foreground face can be
selected when its area is at least four times the next face. Comparable people and
blur still cause abstention. The retry classifier was retrained on this exact input
pipeline. Neither a scored standard result nor its uncertainty decision is overwritten.
Both use frozen, verified EfficientNet-B0 features and fitted compact classifiers.

## Data and fitting

638 balanced owner-labeled candidates (319/class) come from 822 unique files; 184 extra
Fake files remain reserved. Augmentations are not independent sources. Original/fake
families and augmentations stay in one split. Identity/donor independence is unverified.

| Component | Quality eligible | Train Real / Fake | Validation Real / Fake | Calibration Real / Fake |
|---|---:|---:|---:|---:|
| Standard | 593 | 177 / 177 | 11 / 11 | 11 / 11 |
| Retry | 616 | 179 / 179 | 11 / 11 | 12 / 12 |

Standard selected files: 432; retry selected files: 440.
Only training includes augmented files. Component reports and full selection records
are in reports/standard_training and reports/retry_training. Each classifier was selected
using validation ROC-AUC and validation Brier tie-breaks, with calibration fitted on
separate originals. The retry classifier's common-validation AUC was 0.760 versus 0.727
for standard. It regressed as a global replacement (65% on the reused reserved pool),
so it is used exclusively for insufficient-evidence retries. This observation is retained
rather than hidden. Its global accuracy is not advertised as an improvement.

## Pipeline regression evidence

The existing reserved files were inspected during development. These measurements are
regression monitoring, not a fresh blinded test or proof of accuracy on unknown videos.

- Quality-eligible candidates: 593/638 before, 616/638 after.
- Reserved originals analyzed: 18/20 before, 20/20 after.
- Accuracy at the 0.5 boundary: 72.2% on 18 before;
  75.0% on 20 after (15 correct).
- All previously scored original-video outputs are unchanged within 1e-7.
- Clear answers: 14/20; accuracy on those answers: 85.7%.
- ROC-AUC 0.840; F1 0.762; confusion matrix [[7, 3], [2, 8]].
- Source-family bootstrap accuracy interval: [0.6, 0.9].

The thresholds remain 0.35 and 0.65. Unfamiliar features and near-boundary scores still
produce Not sure. Scores are fake-pattern evidence, not probabilities of truth. Face-free
videos still cannot be classified. No arbitrary Real/Fake answer is forced.

## Stress checks on the same 20 reserved originals

| Condition | Analyzed / pool | Accuracy at 0.5 | Clear answers |
|---|---:|---:|---:|
| original | 20/20 | 75.0% | 14 |
| compression | 19/20 | 68.4% | 11 |
| blur | 19/20 | 73.7% | 12 |
| darkness | 20/20 | 75.0% | 11 |

Transformed files are not new independent data. End-to-end timing is recorded in
robustness/summary.json. One representative 24-sample decoding benchmark was 1.848s
with seeks and 0.123s sequentially; it is not a universal or end-to-end speed guarantee.
Only the retry path changes decoder behavior; standard predictions remain identical.

The result screen reports actual face counts and quality reasons and previews frames
across the clip. Windows cleanup and dependency failure paths are tested by simulation;
a real Windows runtime was unavailable. External-video accuracy remains unmeasured.
Model files, config and assets must be updated together. See INSTALLATION.md.
