# Actual video-only evaluation report

## Dataset and separation

The owner supplied 98 user-labeled videos, 49 per class. Structural probing passed for
all 98; file content hashes were unique. Filename-related originals/derivatives remain
in one source group. Seed 42 fixes 58 train, 12 validation, 12 calibration and 16 test
inputs. Identity/donor metadata was absent: this is **not a certified identity-disjoint
or externally sourced benchmark**. Dataset origin and labels were not independently verified.

Quality filtering retained 73 videos: 42 train, 9 validation, 12 calibration,
10 test. All 25 excluded videos and reasons are in reports/results/excluded.csv. The
reserved test denominator remains 16; exclusion is not success. Coverage is 62.5%.

## Primary result: raw visual classifier score at a fixed 0.5 threshold

| Metric | Actual measured value |
|---|---|
| Evaluated videos / source groups | 10 / 5 |
| Accuracy / balanced accuracy | 60.0% / 60.0% |
| Precision / recall / F1 | 0.600 / 0.600 / 0.600 |
| ROC-AUC | 0.800 |
| PR-AUC (trapezoid) | 0.846 |
| Average precision (distinct metric) | 0.858 |
| False positive / false negative rate | 40.0% / 40.0% |
| Brier score / 10-bin ECE | 0.2828 / 0.3177 |
| Source-pair bootstrap accuracy interval | 50%–80% |

Confusion matrix, rows=true and columns=predicted, order Real/Fake: [[3, 2], [2, 3]].
Six correct and four incorrect classifications occurred among ten analyzed test videos.
Intervals use 1,000 source-pair bootstrap draws and are unstable with only five source
groups. Subject independence is not established. A high ranking AUC does not erase
wrong decisions or establish perfect accuracy.

## Fair comparisons

| Model | Held-out accuracy | ROC-AUC |
|---|---|---|
| EfficientNet transfer classifier | 60% | 0.80 |
| texture_logistic | 70% | 0.64 |
| texture_forest | 60% | 0.56 |

Candidates for the main classifier were chosen using validation only. Baselines used
training-only features/scaling and the same test sources. We did not switch the deployed
model to whichever scored best on test. Raw and temperature-scaled outputs are retained
separately; the small calibration fit did not establish trustworthy confidence. The app
therefore presents raw scores as scores, not probability of truth.

## Actual upload-route robustness evaluation

| Picture condition | Analyzed / reserved | Accuracy at 0.5 | Clear app answers |
|---|---|---|---|
| original | 10/16 | 60% | 9 |
| compression | 10/16 | 50% | 8 |
| blur | 10/16 | 70% | 9 |
| darkness | 10/16 | 60% | 9 |

These are fixed derivatives of held-out source videos, **not independent external data**.
Compression uses H264 CRF35, blur Gaussian sigma1.2, and darkness brightness -0.12.
All conditions use unchanged model parameters. Mean complete check time was about
1.34 seconds on original inputs in this environment. These
latencies are measurements on short local videos, not guarantees for arbitrary lengths.
The app also uses an uncertainty band and quality/distance gates; its clear-answer
coverage/accuracy are reported separately, not substituted for the 0.5 benchmark.

## Reproducible evidence and practical limits

Actual JSON/CSV files, per-file predictions, comparison curves and robustness outputs
are in reports/results. checkpoints/visual.json is a real fitted 201,871-byte head at
this run; the frozen official backbone is approximately 21.4 MB. Source and artifact
hashes are recorded. The full feature extraction/first fit took about 137 seconds;
a cached refit took about 5.61 seconds.

Unit and integration test success verifies software contracts, not universal unknown-video
accuracy. Labeled external videos were not supplied or evaluated. This model is suitable
for demonstrating the workflow and its failures, not for reliable real-world decisions.
