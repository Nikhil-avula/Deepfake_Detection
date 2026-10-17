# Research definition and feasibility

**Problem:** determine whether combining facial, voice, temporal and synchronization
evidence improves held-out manipulation detection, especially when one modality is absent
or unreliable. An overall fake label and a modality fake label are different targets.

**Primary contribution:** quality- and availability-aware late fusion, evaluated against
simple averaging and unimodal models on identical splits. Existing multimodal datasets
and detectors already combine modalities; this project does not claim multimodality,
EfficientNet, LSTM, Grad-CAM or an interface is itself novel. Any novelty is the specific
controlled experimental analysis of quality, missingness and disagreement. A formal
prior-work review and measured comparison are required before a publication novelty claim.

## Questions, hypotheses and success criteria

| Question | Testable hypothesis | Test |
|---|---|---|
| Does fusion add value? | Quality-aware fusion improves video ROC-AUC over averaging | Same independent test groups, group-bootstrap intervals |
| Does quality help? | Quality covariates improve Brier score under degradation | Fixed compression/blur/lighting/noise perturbations |
| Does temporal evidence help? | LSTM adds value beyond static frames | Same visual encoder and test videos, inference ablation and retrained ablation |
| Can the system abstain usefully? | Abstention reduces error at lower coverage | Coverage versus selective risk, thresholds chosen without test tuning |
| Does it generalize? | External performance drops less with quality-aware fusion | Separate named external datasets; comparable labels only |
| Is sync meaningful? | Learned aligned pairs separate from within-recording shifts | Real held-out subjects, offset sweep and natural-delay controls |

Success requires reproducible improvements with uncertainty, not a chosen high accuracy
target. Negative findings are valid. Do not pool ASVspoof audio and Celeb-DF visual
numbers into a single incompatible multimodal benchmark.

## Requirements

Functional: authorized import, explicit modality labels, leak-safe splits, four processing
paths, three classical baselines, separate modality models, fusion and calibration,
unseen-data evaluation, interpretable evidence, upload validation, reports and dashboard.
Nonfunctional: honest missingness, bounded decoding, typed modular Python, seeded runs,
provenance hashes, no media redistribution, accessible UI, CPU inference and documented
Windows/Linux setup. Full abstract mapping is in REQUIREMENTS_TRACEABILITY.md.

## Hardware and study size

Observed CPU-only machine: 5 logical CPUs, roughly 33 GiB host RAM and 30 GiB disk.
Host RAM may exceed a container limit; no allocation assumes all of it is available.
Inventory actual licensed file sizes before downloads. Start with a declared stratified
research subset only after grouping identities; label the subset and never equate it with
full-dataset results. Prefer a GPU with approximately 12–24 GiB VRAM for exploratory
training; required VRAM depends on sequence resolution and batch size, so profile first.
Current optimizer uses one video per step and frame chunks of eight. Config `batch_size`
is reserved for future batching and does not alter the current optimization.

## Experiment sequence

1. Validate actual file structure and donor/source metadata after permission arrives.
2. Freeze connected-group splits before tuning; five partitions separate model training,
   model selection, fusion fitting, calibration and final evaluation.
3. Fit baselines using training-only imputation/scaling; train modality networks.
4. Verify synchronization using genuine pairs, shift sweeps and encoding-delay controls.
5. Fit fusion on reserved examples, temperature on calibration, evaluate test once.
6. Run missing-modality and four degradation experiments at fixed severities.
7. Evaluate permitted external visual/audio benchmarks with correct modality labels.
8. Report per-method, per-group and aggregate results; retain failures and coverage.

## Risk register

| Risk | Mitigation / remaining limitation |
|---|---|
| Dataset permission unavailable | Do not download restricted files; continue independent code and tests |
| Identity/source metadata incomplete | Refuse invented metadata; manual verified adapter required |
| Giant connected group | Report infeasible split; redesign dataset selection with advisor |
| CPU/storage insufficient | Declared subset or external GPU, never relabel subset performance |
| Haar face bias/occlusion | Report missingness, abstain; evaluate a stronger licensed detector later |
| Energy is not speech detection | Mark as usability heuristic; evaluate VAD before claims about speech |
| Sync learns codec shortcuts | Within-recording negatives and held-out identities; add delay controls |
| Quality covariates encode dataset identity | Cross-dataset and quality ablations |
| Overconfidence/domain shift | Calibration metrics, coverage and explicit limitations |
| Privacy and redistribution | Local-only authorized media, no tracked raw data or user uploads |

Development authorship is AI-assisted in this session. Human team roles and experiments
must be filled by the actual team; no contributions are invented.
