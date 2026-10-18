# What this pilot cannot promise

- The main classifier achieved 60% accuracy on ten analyzable held-out videos. It is
  not a reliable general-purpose authenticity service or a perfect college benchmark.
- Six of sixteen reserved test videos were rejected for insufficient/ambiguous facial
  evidence. Coverage and errors are reported separately; exclusions are not correct predictions.
- Labels came from the owner. Origin, manipulation method, participant consent and
  donor/subject identities were not independently verified.
- Source pairs are separated, but identity/donor-disjoint testing is unverified.
- Ninety-eight videos provide limited diversity. Modern unseen generation methods,
  different populations, lighting, pose and video processing can cause failures.
- Haar detection can miss profiles or produce false boxes. Blurry or multi-person inputs
  may be refused, including otherwise legitimate videos.
- ImageNet features are frozen; the trained head can learn identity or dataset shortcuts
  rather than general forgery evidence. Feature-distance gating does not solve this.
- Model scores are not calibrated probabilities of truth. A separate temperature fit
  was evaluated and does not justify a confidence guarantee.
- Fixed score thresholds and quality gates are engineering policies. No validated
  operational false-alarm target has been established.
- Robustness derivatives of the held-out uploads are not independent external videos.
  No labeled, newly sourced external corpus has been evaluated.
- Clear answers can still be wrong. Human/source verification remains necessary.
- Temporary files use normal deletion, not certified secure erasure. Session evidence
  persists until cleared or disconnected. No raw media are included in Git by default.
- Windows setup is documented but was not executed on Windows. There is no real-time,
  pixel-level forgery localization or identity-verification claim.
