# Abstract and request traceability

Status vocabulary: **implemented/tested** means the named engineering check ran;
**implemented/unvalidated** means code exists without a real dataset experiment;
**blocked** requires data/assets/compute; **pending** means work is not implemented or
executed yet. No row implies trained performance without an artifact. The authoritative
abstract has eight substantive paragraphs, all mapped below. Extended user-request
requirements follow in the second table.

| ID / abstract requirement | Code or design | Experiment / test and evidence | Status |
|---|---|---|---|
| A1: AI-generated faces/voices and misuse motivate detection | README; docs/RESEARCH_PLAN | Motivation documented, not an empirical claim | Documented |
| A2: Detect face, voice, frame and sound–lip manipulations; generalize to new methods | models.py; media.py; inference.py | Modality paths and tensor/gradient tests; actual unseen-method evaluation absent | Implemented/unvalidated; data blocked |
| A3: Joint visual/audio/temporal/lip-sync system | Visual, Audio, Temporal, Synchronizer; fusion.py | tests/test_models.py; held-out multimodal experiment pending | Implemented/unvalidated |
| A3: Real/Deepfake output, confidence and facial/audio/sync breakdown | Analyzer.analyze; app/main.py Results | Missing-model abstention tested; calibrated checkpoint analysis on genuine file unrun | Partial; data blocked |
| A4: Security/verification application context | README; About page; limitations | Documentation review | Documented, no deployment claim |
| A5: OpenCV frame extraction and face crop, separated audio | media.preprocess, extract_audio | Actual generated MP4 decoded with FFmpeg, no-face/audio/missing-audio tests | Implemented/tested engineering path; genuine faces pending |
| A5: CNN/EfficientNet or ResNet visual detector | models.Visual, training.train | EfficientNet shape, gradients and Grad-CAM tests | Implemented/unvalidated training |
| A5: Mel-spectrogram audio CNN | media.mel, models.Audio | Real waveform extraction, feature shape, optimizer update tests | Implemented/tested engineering; spoof accuracy blocked |
| A5: LSTM or Transformer temporal analysis | models.Temporal; frozen Visual.embed | Sequence tensor test; trained static/temporal comparison pending | Implemented/unvalidated |
| A5: Compare mouth and audio synchronization | MouthLandmarks, Synchronizer, sync_inputs | Explicit offset/gap tests; asset download denied; real alignment study pending | Asset/data blocked |
| A5: Multimodal fusion | fusion.py; reliability_fusion | Weight/missingness tests; fusion fitting and benchmark pending | Implemented/unvalidated |
| A6: Logistic Regression, Random Forest, XGBoost first | training.baseline | Train-only sklearn pipeline; actual held-out baseline runs pending | Implemented/unvalidated |
| A6: Consider FF++, Celeb-DF, DFDC, FakeAVCeleb, ASVspoof | DATASET_GUIDE | Official access observations and uncertainty recorded | Assessed; no data acquired |
| A6: Python scientific stack and upload Streamlit UI | requirements; app/main.py | Installed environment, eight-page AppTest | Implemented/tested |
| A7: Unseen methods and cross-dataset generalization | RESEARCH_PLAN; explicit manifests and predict CLI | External overlap/label audit and test results absent | Blocked on authorized data |
| A7: Accuracy/precision/recall/F1/ROC-AUC/PR-AUC/CM/FPR/FNR | evaluation.score/evaluate | Known-answer metrics and generated plots tested | Implemented/tested formulas; research outputs absent |
| A7: Compare unimodal and fusion robustness | predict/evaluate/fusion-evaluate/perturb | Executable commands; actual fair study pending | Implemented/unvalidated |
| A8: Working system with supporting scores and unseen-data study | CLI, dashboard, reporting | Engineering workflow verified; real checkpoint demonstration absent | Partial; not final research acceptance |
| A8: Compression/noise/lighting/language limitations, no universal detector claim | LIMITATIONS; quality gates | Abstention tests; bias/language evaluation pending | Documented and partially implemented |
| A8: Future real-time, explanation and continuous learning | Grad-CAM available; future scope documented | Attribution spatial test; real-time/continuous learning intentionally future | Grad-CAM tested; future scope unclaimed |

## Extended request audit

| Request area | Artifacts / tests | Remaining gap |
|---|---|---|
| Research questions/hypotheses/novelty/risks/diagrams | docs/RESEARCH_PLAN; ARCHITECTURE | Formal broader literature review and measured novelty evidence |
| Thirteen datasets, licensing/size/access comparison | DATASET_GUIDE | Several official releases unverified; actual licenses/files unavailable |
| Download/resume/checksum | data.download | Official checksums/access and network-resume integration test pending |
| Inventory/corruption/class counts/provenance | data.inventory; utils.sha256 | Dataset-specific adapter, full packet-corruption study and actual counts |
| Source/identity leakage-safe splits | data.split_manifest; CLI hashes; connected-group tests | Accurate donor/target/source metadata required; stratified subset helper pending |
| OS/CPU/GPU/RAM/disk/pins/Windows/Linux | INSTALLATION; environment records | Windows execution unverified; GPU not present |
| Face crop/track/normalize/sequences/quality/cache | media.preprocess, Visual.embed | Haar limitations; robust tracking and actual faces not yet validated |
| Audio mono/rate/Mel/silence/clipping | media.extract_audio/mel; integration tests | Energy gate is not trained VAD; speech/music discrimination pending |
| Mouth landmarks/PTS alignment/FPS/offset | MouthLandmarks; audio_window; sync_inputs tests | Landmark asset network denied; actual landmark and VFR-media tests pending |
| Augmentation/imbalance/loops/scheduler/early stopping/checkpoints | training.train | Authorized real training, optimizer histories and best artifact absent |
| Grad-CAM, spectrogram and temporal interpretation | explain.py; dashboard | Real trained attribution usefulness not evaluated |
| Real sync network and held-out pair validation | Synchronizer; sync-evaluate CLI; shifted-input and gradient tests | Real shift-sweep evaluation pending; no validated lip-sync classification claim |
| Individual/simple/learned/reliability fusion | models.py; fusion.py; score-only comparator, modality dropout and integration tests | Calibrated quality-aware study on real data pending |
| Missing/low-quality/multiple/conflicting evidence | Analyzer; abstention and weight tests | Genuine checkpoint integration and selective-risk curve pending |
| Calibration | fit_temperature; reserved calibration split | Temperature fitting alone not validated confidence; held-out ECE/Brier required |
| Metric suite, intervals, latency/size/memory | evaluation; predict sidecars | Group-bootstrap accuracy only; per-model runtime/memory profiling pending |
| Ablation/missing modality/cross-dataset/degradations | fusion.assess; robustness.perturb | Inference ablations implemented, retrained ablations and actual external/degraded experiments pending |
| Eight-page dark forensics dashboard | app/main.py; styles; AppTest | Genuine trained-checkpoint upload E2E pending |
| Reports, privacy, upload limit and error states | reporting; temp directories; media limits | HTML delivered; PDF optional and not implemented; no promise of forensic secure deletion |
| Unit/integration/E2E tests | tests; execution results in PROJECT_STATUS | Genuine held-out inference necessarily unrun; no research fixtures presented as test corpus |
| Reproducibility/config/seeds/checkpoint provenance | config; utils; training metadata; environment freeze | Actual dataset manifests/splits/curves/weights require data |
| Student documentation/40 viva Q&A/demo/team roles | Required root Markdown documents | Human team must enter actual contributions/results |

The remaining gaps are deliberately visible. This project is not complete under the
user's final acceptance criteria until authorized data, verified landmark processing,
four trained/evaluated modalities, comparative fusion and real checkpoint inference exist.
