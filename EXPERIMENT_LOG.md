# Experiment log

## E000 — Environment and engineering validation, 2026-10-17

- Source baseline: Git `5eeedcf7b91bcf356ed24b1ba56dbaddf6e1cdd6`; new code uncommitted.
- CPU Linux, Python 3.12.14; no detected CUDA. FFmpeg present.
- Installed CPU Torch 2.6.0 and torchvision 0.21.0, pinned research stack and optional
  MediaPipe. Resolved duplicate OpenCV distributions by retaining one pinned contrib
  wheel. Full package environment recorded separately.
- Official FakeAVCeleb request instructions verified. User confirmed no approval yet.
  Landmark model host denied by proxy. No restricted data or unverified weights obtained.
- Initial `pytest -q`: 23 passed in 23.88s. These are engineering checks only.
- No optimization on real deepfake data and no detection metrics produced.
- Exact final validation commands, failures/fixes and subsequent outcomes: PROJECT_STATUS.

## Future run template

Record: ID/date/operator; research question; dataset release/permission/provenance;
inventory/split hashes and group rules; code commit + dirty patch; complete config;
hardware/runtime/dependencies; random seed/nondeterminism; exact command; start/end time;
training/validation counts and exclusions; checkpoint hash; calibration procedure;
held-out prediction/metric artifacts; coverage; failures; interpretation and next test.
Do not overwrite negative runs or label a subset as a full-dataset experiment.
