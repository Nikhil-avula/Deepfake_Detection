# Model reliability audit — 19 October 2026

The current model is a research pilot. It does not yet establish reliable detection on unknown videos. Successful software tests or startup checks do not establish classification accuracy.

## What “Not sure” means

Analysis completed. The standard configuration returns this decision when the raw fake-pattern score is between 0.35 and 0.65, or when the feature-distance check marks the input unfamiliar. Even a score outside that interval can be unfamiliar. These scores are not probabilities of authenticity.

Repeated uncertainty is a model limitation, not evidence that the upload failed. Forcing every score above 0.5 to Fake and every lower score to Real would conceal uncertainty without improving the learned model. No decision thresholds were weakened in this update.

## Recorded evidence

From `reports/standard_training/all_predictions.csv`, using a 0.5 binary threshold:

| Partition | Correct / evaluated | Accuracy |
| --- | ---: | ---: |
| Training | 347 / 354 | 98.0% |
| Validation | 14 / 22 | 63.6% |
| Calibration | 15 / 22 | 68.2% |

The gap suggests overfitting. The frozen ImageNet EfficientNet encoder and small fitted classifier have learned patterns that do not transfer consistently. The data contain related augmented examples; more such files do not provide the diversity of independent sources or people. Identity independence has not been established.

The deployed pipeline's recorded 20-video original-source regression check (`reports/results/robustness/original/predictions.csv`) has 15/20 correct at a forced 0.5 threshold. The actual UI gives 14 clear answers, of which 12 are correct, and six Not sure results. This pool was inspected during development and is not a fresh blind test.

Known wrong clear predictions include `v14.mp4` (provided Real; Likely fake, score 0.718) and `vs9.mp4` (provided Fake; Likely real, score 0.303). No filename-based corrections or test-label tuning have been applied.

## Changes and remaining work

The Windows launcher and `doctor` command check the execution environment and model assets before startup. The result screen now explains the two reasons for uncertainty. Neither change improves learned classification accuracy; the model weights remain unchanged.

Reliable improvement requires diverse, independently labeled original real and manipulated videos, source/identity grouping before splitting, and augmentations confined to training. Model changes must be selected using validation data, with final accuracy, class recall, false positives and uncertainty coverage measured once on a new blind set. Keep an external-source set to assess transfer to unfamiliar videos.

There is no justified accuracy guarantee or guaranteed dataset count from the current evidence. The present model should not be presented as a dependable authenticity checker.
