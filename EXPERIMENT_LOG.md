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
