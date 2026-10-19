# Limits of the current detector

- Current measured accuracy and coverage are reported in reports/results/summary.json and EVALUATION_REPORT.md. No result establishes accuracy on arbitrary unfamiliar videos.
- The balanced candidate dataset has 102 original files and 217 augmented files per class. Augmented variations are correlated evidence, not additional independent sources.
- Labels were supplied by the owner. Dataset provenance, participant identities, donor relationships and label correctness were not independently verified.
- Grouping uses conservative filename relationships. Historical assignments are preserved; identity-disjoint evaluation is not certified. Newly named sources are not necessarily unseen identities or an external benchmark.
- Quality exclusions and equal-count selection reduce the final number used for learning and evaluation. Coverage must be read alongside accuracy. Augmented held-out variants are robustness checks, not new independent tests.
- Face localization can miss profile, occluded, small or blurry faces. Multiple people and insufficient suitable samples lead to Cannot analyze. The model examines faces rather than every possible type of video manipulation.
- Container compression, resolution, lighting or source-specific artifacts may correlate with labels. Realistic testing across independent sources and generation methods is needed to quantify these biases.
- EfficientNet is a frozen ImageNet encoder; only a compact classifier is fitted. A neural candidate is selected only when its validation performance justifies it.
- The raw score is not a probability of truth. Temperature calibration on a small separate split may be unstable. The feature-distance uncertainty gate is heuristic.
- Results are research evidence, not proof of authenticity or manipulation. The project does not promise perfect accuracy, universal detection or guaranteed performance on an unknown upload.
