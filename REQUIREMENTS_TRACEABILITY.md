# Updated requirement traceability

The authoritative scope is the owner's updated video-only statement and latest request
for revised files, understandable UI, genuine training, honest evaluation and testing.
The earlier project specification is superseded. No excluded component remains an active
requirement or implementation dependency.

| Requirement | Implementation | Actual verification / limit |
|---|---|---|
| Upload a video and inspect visual facial evidence | app/main.py; media.py; inference.py | Real held-out input-route checks and UI tests |
| Use the owner's real/fake uploads | data.import_uploads | 98 files downloaded, probed and hashed; labels supplied by owner |
| Keep related inputs out of different splits | data.split_pairs/assert_no_leakage | Matching source pairs and identical content grouped; unit checks; identity metadata unavailable |
| Train a genuine visual classifier | models.py; training.py; checkpoints/visual.json | Actual frozen-backbone features and fitted decision head; not full backbone fine-tuning |
| Select settings without test leakage | training.py | Train-only scaling and validation-only candidate selection; separate calibration/test |
| Clear real/fake result with uncertainty | Analyzer; show_result | Likely real/fake, Not sure and Cannot analyze; no fake fallback score |
| Accessible polished UI for nontechnical users | app/styles/theme.css; six-page navigation | All page render tests; plain language, actual results, clear/reset and report download |
| Explain input evidence | results face samples and frames | Actual extracted images; no fabricated forged-region heatmap |
| Honest measured evaluation | evaluation.py; reports/results | Metrics, source-pair intervals, comparisons, errors, exclusions, score variants and coverage |
| Unknown-input failure behavior | test_media.py; test_model_ui.py; experiments.py | Corrupt/empty/non-face/renamed/changed-quality media tested; real unfamiliar accuracy remains unverified |
| Reproducibility | configs; scripts; JSON head; manifests/hashes | Exact setup/train/check/start commands; no raw media distribution |
| Privacy and robust errors | temp directory cleanup; escaped HTML; upload limits | Tests for invalid media, size/duration limits, HTML escaping and clear session |
| Perfect accuracy on arbitrary unknown data | Not an achievable software guarantee | Actual accuracy 60% on 10 analyzed test videos; more independent data required |

Implementation and test completion do not promote the final row to success. Current
capabilities are a trained visual research pilot and a working interface, with explicit
limitations, not a universally validated detector. Full command evidence: PROJECT_STATUS.
