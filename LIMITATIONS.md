# Limitations and interpretation

- No trained detectors or authorized data are present. This is implemented research
  software with engineering checks, not a measured authenticity service.
- Dataset-specific metadata parsing is deliberately not guessed. Missing donor/target
  identities can still cause leakage even if code passes its checks.
- Haar face detection is sensitive to pose, skin/lighting variation, occlusion and small
  faces. IoU tracking can lose a moving subject; multiple faces cause abstention.
- Uniformly sampled frames and conservative gap rejection may discard useful dynamics.
  Variable-frame-rate and complicated edit-list timestamps need additional validation.
- Audio RMS gating is not speech recognition or a validated voice activity detector.
  Music/noise can pass it. Clipping/quality thresholds are engineering heuristics.
- Landmark task asset access is blocked. The library is installed but actual landmark
  extraction cannot yet be validated. Mouth windows are absent until the asset exists.
- Learned synchronization has no trained weights or real held-out shift validation yet.
  Dubbing, language, encoding offsets and natural production delay are confounders.
- Quality-aware fusion may learn dataset/codec shortcuts. Its benefit is a hypothesis.
  A score-only learned comparator and modality-dropout training are implemented but
  unvalidated on real data.
- Calibration and policy thresholds do not guarantee probability of truth, especially
  under unseen generation methods, language, compression, blur or noise.
- Grad-CAM indicates influential image regions, not a proven forged-region mask.
- Generated engineering media test plumbing, never research performance. Full-dataset,
  cross-dataset, minority-group and robustness claims are all unavailable.
- Windows/GPU execution is documented but not verified on this CPU Linux machine.
- Files are deleted through ordinary temporary-file cleanup; this is not certified
  forensic secure erasure. Session memory holds extracted evidence until cleared.

Real-time inference and continuous learning are future directions in the abstract, not
completed capabilities. Use human verification and trusted independent evidence for
important decisions; do not rely on an automated research score alone.
