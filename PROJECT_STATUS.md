# Video-only project execution status — 2026-10-18

## Windows setup diagnosis and correction

The owner's screenshot showed WinError2 before any face samples and Model training
pending. The launch of ffprobe failed before model inference; the prior exception
handler mislabeled machine setup failure as unsuitable video evidence. In addition,
the previous repository excluded the local trained head, leaving clones without it.

Added explicit FFprobe dependency errors, a distinct Setup required outcome, preflight
UI messages and a disabled analysis action until prerequisites exist. The small fitted
checkpoints/visual.json is now included in Git (201,871 bytes; same verified trained
artifact). Raw videos and feature caches remain excluded. Windows installation and
terminal/PATH restart steps are documented. No additional dataset or retraining is
needed to fix this setup failure. Classifier parameters/evaluation results are unchanged.

New regression tests distinguish missing executable/model setup from invalid media;
real-data UI tests also check for private manifests before running in a fresh clone.
The Linux suite passed35 tests; Windows PATH failures are simulated, not claimed as
execution on an actual Windows machine. Exact final outcome is in experiments/validation.txt.

## Current problem statement

A user uploads a video. The application analyzes visual facial evidence and gives a
plain-language real/fake assessment, with Not sure or Cannot analyze when appropriate.
The owner explicitly replaced the previous scope and authorized revising files, using
their uploaded dataset, training a model and testing the application. All active code,
configuration, dependencies and documentation now describe this video-only workflow.

## Work actually completed

- Preserved the previous checkout and uncommitted changes in the verified archive
  /workspace/deepguard-backups/pre-video-only-20261018.tar.gz (284 archive entries).
- Downloaded all 98 authorized uploads: 49 real-labeled and 49 fake-labeled videos,
  about 147.35 MiB. Probed every video; zero basic decode failures and 98 unique hashes.
- Fixed 49 source-pair groups using matching filename stems; byte-duplicate and label
  checks are implemented. Identity/donor-disjoint status remains unverified.
- Split with seed42 into 58 train, 12 validation, 12 calibration and 16 test files.
- Extracted real face features using a frozen official ImageNet EfficientNet-B0 encoder.
  The official backbone cache passed SHA-256 verification. No generated training data,
  memorized-label lookup or fake confidence score was introduced.
- Trained a regularized standardized classifier head using 42 quality-eligible training
  videos. Selected C on 9 validation videos only. Fitted/evaluated a separate temperature
  transformation on 12 calibration videos. Raw scores remain the user-facing evidence.
- Measured actual held-out accuracy: 60% on 10 analyzable videos from 5 source groups;
  6 of16 reserved videos were quality-rejected. Coverage62.5%, precision/recall/F1 .60,
  ROC-AUC .80, confusion matrix [[3,2],[2,3]]. No perfect/general unknown-data claim.
- Trained/evaluated two texture-only baselines and recorded per-video predictions,
  metrics, source-pair bootstrap intervals, exclusions and calibration variants.
- Ran the real upload inference route on original held-out sources and fixed compression,
  blur and darkness variants. Accuracy at .5:60%,50%,70%,60%; each analyzed10/16 files.
  These are existing-source robustness inputs, not an independent external benchmark.
- Rebuilt the interface with six plain-language pages, actual evidence, score explanations,
  visible reliability limitations, model/dataset information, privacy reset and HTML reports.
- Rewrote all project learning/install/training/evaluation/research documents for the
  current statement. Replaced previous unrelated modules, scripts and tests.

## Actual commands and evidence

From /workspace/Deepfake_Detection in /workspace/deepguard-venv:

```bash
deepguard import-uploads /tmp/deepguard_uploads.json
deepguard train data/manifests/splits.csv
deepguard robustness
bash scripts/setup.sh
bash scripts/validate.sh
bash scripts/run.sh
```

The import was executed through the equivalent Python import_uploads/split_pairs functions.
The local attachment index is not published. The first full feature pass/fit took about
137seconds; a cached refit took5.61seconds. Runtime is CPU-only, with four-core quota and
32GiB limit. The fitted head is about202KB and backbone about21.4MB, retained locally.

Final suite: **33 passed, 15 upstream warnings in14.10s**, no failed/skipped/expected-fail
cases in this machine. Both real-data integration tests executed. Cases include leakage,
metrics, tensor shapes, decode limits, corrupt/empty/non-face files, actual held-out
inference, unchanged predictions after filename renaming, report escaping, clear-session
behavior and all six actual UI routes. These tests prove software behavior, not correct
classification of every unfamiliar video. Test log:experiments/validation.txt.

Setup refresh and pip check succeeded. Streamlit was restarted on port8501. Local health
returned HTTP200 with body ok; root returned HTTP200. AppTest verified the six pages and
an actual inference result. No unsupported loopback preview link is supplied.

Runtime versions and code hashes: experiments/runtime.json and experiments/source_hashes.json.
Held-out split membership, predictions, plots and robustness: reports/results.

## Failures, corrections and limits

An initial rule rejected a whole clip after any ambiguous face sample. During preprocessing
this was corrected to skip ambiguous frames and reject over25% ambiguity, before final
held-out results were inspected. Feature caches were regenerated. Twenty-five videos
still fail the declared quality gates, and all exclusions remain reported.

Temperature scaling on the small calibration set can become overconfident; raw/scaled
variants are retained and the app never labels its score as guaranteed confidence. The
60% result is not adequate for reliable real-world deployment. More diverse licensed,
independently labeled sources and identity/donor metadata are needed for stronger claims.

One setup record command initially used the system Python without the package; it was
rerun successfully in the actual project virtual environment. Startup initially found
port8501 occupied by this project's earlier server; that known process was stopped and
the new video-only service was launched. No foreign service or user files were removed.

## Remaining limitations

- No verified independent external dataset was supplied or tested.
- Perfect manual unknown-video accuracy cannot be guaranteed; false results are recorded.
- Raw files and feature caches remain ignored. Following the Windows setup diagnosis,
  the small trained JSON head is included in Git so clones can run inference without
  retraining. FFprobe must be on PATH; the generic backbone downloads if not cached.
- Origin/permission and participant identity metadata must be confirmed by the owner.
- Windows/GPU execution is documented where relevant but was not tested here.
- Saved cloud setup/startup guidance is refreshed for the new scope; saving is not
  publication or proof of fresh-task restoration.
