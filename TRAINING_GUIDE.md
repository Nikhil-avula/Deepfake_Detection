# Training and reproducibility

Run from the repository root in the installed environment. These are executable commands,
but **dataset training has not been executed** because authorized data are unavailable.
Never call engineering fixture results detection performance.

```bash
deepguard hardware
deepguard inventory data/manifests/input.csv data/manifests/inventory.csv
deepguard split data/manifests/inventory.csv data/manifests/splits.csv
deepguard prepare data/manifests/splits.csv --landmark-asset checkpoints/face_landmarker.task
deepguard baseline data/manifests/splits.csv reports/results/baselines
deepguard train visual data/manifests/splits.csv checkpoints/visual.pt
deepguard train audio data/manifests/splits.csv checkpoints/audio.pt
deepguard train temporal data/manifests/splits.csv checkpoints/temporal.pt --visual-checkpoint checkpoints/visual.pt
deepguard train sync data/manifests/splits.csv checkpoints/sync.pt --landmark-asset checkpoints/face_landmarker.task
deepguard sync-evaluate data/manifests/splits.csv checkpoints/sync.pt checkpoints/face_landmarker.task reports/results/sync-pairs
```

All modality paths require correct labels and usable examples in train and val.
Visual/audio/temporal require both target classes. Sync uses real-face/real-audio clips
and generates matched/shifted examples within each clip; this task is separate from
overall deepfake classification. No pretrained manipulation detector is included.
`pretrained: true` optionally obtains official ImageNet EfficientNet initialization,
not deepfake-trained weights. Default false avoids hidden downloads. ImageNet weights
come from torchvision's official verified route and require its host access.

Training uses seeded Python/NumPy/Torch, deterministic algorithms where supported,
AdamW, class-weighted binary cross entropy, bounded frame chunks, gradient clipping,
ReduceLROnPlateau, validation-based early stopping and best-weight saving. GPU operators
can still be nondeterministic; warnings must be recorded. One video is optimized per
step. `batch_size` is reserved and currently unused; do not describe it as effective batch
size. Configurable flips/brightness affect visual training; log-Mel perturbation augments
audio features but does not replace waveform-noise robustness evaluation.

Checkpoint metadata includes model kind, training/validation counts, config, manifest
hash, original Git commit/dirty status, runtime versions, and dependent encoder/landmark
hashes. Keep the manifest immutable once dependent models are trained. Use a new run ID
and output directory for each configuration; do not overwrite a published experiment.
Histories are saved beside checkpoints. Checkpoints use tensor-only safe loading;
never load arbitrary pickle files from uploaded media or untrusted sources.

## Fusion

Base models train only on `train`, select on `val`. `fusion` remains unused by their
optimizer. `calibration` fits only the final fusion temperature. `test` is untouched by
fitting/selection. If separately calibrating base models, use an additional reserved
partition rather than reusing final-fusion calibration and document that study change.

```bash
deepguard collect data/manifests/splits.csv fusion experiments/fusion-train.csv --landmark-asset checkpoints/face_landmarker.task
deepguard collect data/manifests/splits.csv calibration experiments/fusion-calibration.csv --landmark-asset checkpoints/face_landmarker.task
deepguard fusion-fit experiments/fusion-train.csv experiments/fusion-calibration.csv checkpoints/fusion.json
deepguard collect data/manifests/splits.csv test experiments/fusion-test.csv --landmark-asset checkpoints/face_landmarker.task
deepguard fusion-evaluate experiments/fusion-test.csv checkpoints/fusion.json reports/results/fusion
```

Quality-aware learned fusion includes score, quality and availability features. A
**score/mask-only learned comparator** is fitted on identical examples and saved alongside
the fusion JSON; its held-out comparison remains unrun. Averaging and reliability-weighted
log-odds are implemented comparison baselines; sync is excluded from direct averaging.
No evidence is replaced with a reported fabricated score. Missing features use neutral
internal placeholders accompanied by availability zero.

Fusion fitting augments training rows by dropping each available modality in turn while
retaining at least one evidence source; the score-only comparator uses identical rows.
These internal augmented rows are recorded separately from observed sample counts.
Test evaluation emits coverage/risk at predeclared abstention bands. These curves are
descriptive held-out outputs, not a license to choose thresholds on the test set.

## Individual predictions, evaluation and optional calibration

```bash
deepguard predict data/manifests/splits.csv checkpoints/visual.pt experiments/visual-test.csv
deepguard evaluate experiments/visual-test.csv reports/results/visual
deepguard predict data/manifests/splits.csv checkpoints/audio.pt experiments/audio-test.csv
deepguard evaluate experiments/audio-test.csv reports/results/audio
deepguard predict data/manifests/splits.csv checkpoints/temporal.pt experiments/temporal-test.csv --visual-checkpoint checkpoints/visual.pt
deepguard evaluate experiments/temporal-test.csv reports/results/temporal
```

For a separate base-calibration experiment, produce `--split calibration` predictions,
then `deepguard calibrate ORIGINAL.pt CALIBRATION.csv CALIBRATED.pt`. Calibrate before
training temporal/fusion dependents, because they verify exact upstream hashes. Do not
replace a visual checkpoint after training its temporal encoder dependency.

## Robustness and external testing

```bash
deepguard perturb authorized-input.mp4 data/interim/compressed.mp4 compression
deepguard perturb authorized-input.mp4 data/interim/blurred.mp4 blur
deepguard perturb authorized-input.mp4 data/interim/darkened.mp4 lighting
deepguard perturb authorized-input.mp4 data/interim/noisy.mp4 noise
```

Create a derivative manifest retaining original group IDs and split membership, then
evaluate the unchanged checkpoints. Use new output paths and preserve paired source
relationships. Do not rerun random splitting for perturbations. Compression CRF 38,
Gaussian blur sigma 1.5, brightness -0.12 and fixed-amplitude additive audio noise are
the declared initial severities. Quantify actual SNR before SNR-specific claims.
External visual/audio datasets need their own verified label/provenance adapters. No
cross-dataset result is available yet; never tune using their held-out labels.
