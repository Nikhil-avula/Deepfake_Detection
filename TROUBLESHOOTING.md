# Troubleshooting

| Symptom | Diagnosis and next action |
|---|---|
| `Model not trained` | Expected without `checkpoints/{visual,audio,temporal,sync}.pt`; obtain data and run genuine training |
| No final classification | Check calibrated `fusion.json`, quality gates and disagreement; do not invent a fallback score |
| FFprobe/FFmpeg not found | Install official FFmpeg build/package and ensure both programs are on PATH |
| Corrupt/unsupported upload | Review codec/duration/dimensions/size; inspect FFprobe locally without uploading private media elsewhere |
| No face / many faces | Haar baseline has limitations; supply suitable authorized footage; multi-person input causes abstention |
| No audio windows | Audio missing, too short for full windows, or below energy gate; this is not a fake label |
| No mouth windows | Need MediaPipe library, authorized FaceLandmarker task asset, face and usable audio |
| Landmark download 403 | Proxy policy denied official host; add `storage.googleapis.com` in environment settings, then retry official download |
| Dataset inaccessible | Complete owner approval first; review exact download host and license, never bypass access controls |
| `Split leakage` | Correct donor/target/source metadata and regenerate a declared split, rather than disabling the check |
| Too few connected components | Identities connect too many files; redesign data selection with the advisor, preserving group safety |
| Both classes required | Inspect per-modality labels and split counts; do not substitute overall labels or test-set examples |
| Encoder/fusion hash mismatch | Upstream weights changed; retrain/recollect dependent models with recorded provenance |
| PyTorch tries large CUDA downloads | Install CPU Torch/torchvision first, then requirements; do not install concurrently into one venv |
| Duplicate cv2 distributions | Keep the pinned `opencv-contrib-python` distribution used by MediaPipe; reinstall editable metadata and run pip check |
| Read-only cache warnings | Set `MPLCONFIGDIR` and `XDG_CACHE_HOME` to writable directories; setup scripts do this |
| Safe checkpoint load fails | Only load tensor/primitive checkpoints produced by this project; no unsafe pickle fallback |
| Metric AUC null | A split contains only one target class; AUC is undefined, not zero or perfect |
| Windows activation restricted | Use the venv's Python executable directly or follow your institution's PowerShell policy; do not weaken system security |

Retain exact failing commands/logs in EXPERIMENT_LOG. Retry after a meaningful fix; never
disable TLS verification, alter expected checksums or fabricate a passing result.
