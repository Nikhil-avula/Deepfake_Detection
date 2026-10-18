# Visual detection experiment log

## V001 — Updated scope and owner-data pilot, 2026-10-18

The owner replaced the earlier specification with video-only detection and authorized
processing/training and a complete UI/documentation revision. The previous project state
was preserved outside the repository in /workspace/deepguard-backups/pre-video-only-20261018.tar.gz.
It is historical recovery material, not current scope or an active application dependency.

- Downloaded all 98 uploaded files (49 per class), about 147.35 MiB. No generated training
  media or guessed real/fake labels were used.
- Recorded inventory/hashes; fixed source-pair splits before final results were inspected.
  Identity/donor-disjoint status remains unverified.
- Initial preprocessing treated any ambiguous sampled frame as reason to reject a whole
  clip. Corrected the general policy to skip ambiguous frames and reject over 25% ambiguity,
  before inspecting final held-out outcomes. Feature caches were rebuilt for the policy.
- Frozen official EfficientNet features were extracted; standardized regularized heads
  were selected on validation. A separate temperature fit was tested. Its overly confident
  behavior is retained in evaluation rather than presented as guaranteed confidence.
- Raw scores are the application's displayed evidence. Both raw and scaled test outputs
  are preserved. No test-accuracy optimization or deliberate mislabeled fallback was added.
- Primary test result: 60% on ten analyzed videos; coverage 10/16. Errors remain visible.
  See EVALUATION_REPORT and per-file predictions for details.
- Ran actual inference on all sixteen held-out videos and their fixed compression, blur
  and darkness derivatives. These are robustness tests, not externally independent data.
- Training artifacts, runtime/package/source hashes and tests are recorded in experiments
  and reports. Check PROJECT_STATUS for final commands, test counts and service checks.

Future experiments must record permission/source/labels, grouping metadata, split hashes,
configuration, code version, selection/calibration procedure, real prediction artifacts,
coverage, failures and actual team roles. A new independent external set is needed for
fresh generalization claims; do not repeatedly tune on the current small test set.

## V002 — Confirmed labels and reproducible retraining, 2026-10-18

The owner reuploaded and explicitly confirmed 49 real and 49 fake videos. SHA-256
verification showed all 98 were exact reuploads of the V001 data. They were not
counted as additional examples. The confirmed private manifest was validated and
the same seed-42 filename-pair split was regenerated, without tuning on test results.

The supervised classifier was fitted again using the verified frozen feature cache.
The resulting coefficients reproduce V001: 73 quality-eligible videos, comprising
42 train, 9 validation, 12 calibration and 10 test videos. Twenty-five videos were
excluded by the unchanged face-quality policy. Accuracy remains 60% on 10 analyzed
test videos out of 16 reserved; ROC-AUC remains 0.80. Reusing the same data does not
establish new external generalization evidence.

Added a prepare-dataset command, integrity checks before cached training, and an
optional confirmed-manifest argument to scripts/train.sh. Added rejection tests for
changed bytes, invalid labels and conflicting class names. The UI now explains
usable/excluded dataset counts and provides current model restoration instructions.
The full suite passed 39 tests; subsequent focused checks passed all 7 model/UI
tests and all 9 dataset tests after their final edits. See experiments/validation.txt
and experiments/retraining_validation.txt for captured results.
