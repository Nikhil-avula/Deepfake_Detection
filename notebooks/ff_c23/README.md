# Train on Kaggle, run locally

Use `DeepGuard_FF_C23_Kaggle.ipynb`. It contains every required code cell, including
an embedded inference module. Do not run `kaggle_training.py` directly: that file is
the editable cell source and its runtime placeholder is filled in the notebook.

## Kaggle steps

1. Open Kaggle → Code → New Notebook → File → Import Notebook. Upload
   `DeepGuard_FF_C23_Kaggle.ipynb`.
2. Choose an available **GPU** in Settings → Accelerator. Enable **Internet**.
   Kaggle may require account verification; availability and quota vary.
3. Choose **Add Input**, search `xdxd003/ff-c23`, and attach that dataset. Attaching
   makes the videos accessible on Kaggle; it does not download 18 GB to your PC.
   Do not run `kagglehub.dataset_download` or copy the videos into `/kaggle/working`.
4. Run the first code cell. Confirm GPU details and writable free space. Leave
   Kaggle's Torch, torchvision and CUDA packages installed as supplied.
5. Run the embedded runtime cell, then the inventory cell. Verify it recognizes
   `original` and the five manipulation methods, with roughly 1,000 files each as
   listed on the data card. Inventory rejects unknown names. If your layout differs,
   inspect `ignored_files.csv` and update the mapping deliberately. Narrow DATA_ROOT
   if multiple versions are attached. DeepFakeDetection is excluded because its
   source/control mapping has not been verified; it is not a valid balanced external
   evaluation using only its fake clips.
6. Run the official-split cell. It downloads tiny JSON metadata, not videos. It
   requires disjoint source-video IDs and excludes cross-partition or unknown pairs.
   Do not randomly split frames or augmented versions. Splits do not establish that
   every human identity is distinct across partitions.
7. Run the assets and face-cache cells. About 5 MB of face detector files are
   downloaded and SHA-256 verified. Up to eight 224×224 JPEG crops are cached per
   video (up to ~48,000 for 6,000 videos). Cache size and extraction time depend on
   the videos; processing is CPU work even with GPU enabled. Failures and method
   coverage are reported. Inspect the displayed TRAIN face crops before training.
8. Run the data-loader and training cells. EfficientNet-B0 ImageNet weights download
   once (~20 MB). The classifier trains for three epochs; final feature blocks then
   fine-tune for up to ten epochs, with early stopping. Mixed precision is enabled.
   Four videos × up to four face crops form each training batch. Use batch size two
   in a fresh run if CUDA memory runs out. Changing settings invalidates old resume
   checkpoints, so select a new WORK folder. Do not install the project's pinned
   local requirements into Kaggle's existing environment.
9. Run validation threshold selection. It selects a binary threshold using validation
   only and places an explicit heuristic uncertainty band around it. No calibrated
   confidence or automatic unfamiliar-input detector is claimed.
10. Run final test evaluation once, after settings are fixed. Keep AUC, balanced
    accuracy, per-method results, false positives, fake recall, face coverage and
    uncertainty coverage. Accuracy among clear answers excludes uncertain videos,
    so it must be shown alongside coverage. The group-bootstrap interval can be
    unstable with few independent source groups. This is in-dataset evidence; test
    separately on external labeled sources before claiming broad generalization.
11. Run the export cell. It verifies one validation prediction against CPU inference
    and creates `/kaggle/working/deepguard_ff_export.zip`. A parity failure blocks
    successful completion and should be investigated rather than ignored.
12. Save a notebook version with outputs. Download the export ZIP through the
    notebook Output/files view. Kaggle sessions and working storage are temporary;
    save checkpoints and cache outputs before a session expires. Merely writing
    `last_checkpoint.pt` does not persist it after the session is discarded.

## Resume after interruption

Within the same live session, rerun cells in order: extraction reuses completed
caches and training loads `last_checkpoint.pt`. At a stage boundary, resume may
start the next stage immediately. Epoch-level resume retains optimizer and scaler
states but is not bit-for-bit identical to uninterrupted training because all
augmentation RNG states are not restored.

For a new session, attach the saved notebook output as an input and copy its
`deepguard_ff` directory into `/kaggle/working/deepguard_ff` **before** running cells.
Keep model, last checkpoint and cache together. Reattach the same dataset/version.
The run fingerprint must match. If paths changed, the extraction cell refreshes
cache source signatures. Preserve original source IDs and partition membership.
Do not move final-test examples into training.

## Local Windows steps (Python 3.12)

Extract `deepguard_ff_export.zip` to a new directory, e.g. `C:\DeepGuardFF`.
In PowerShell in that folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
.\.venv\Scripts\python.exe deepguard_ff_runtime.py "C:\Videos\example.mp4"
.\.venv\Scripts\python.exe -m streamlit run app.py --server.fileWatcherType none --server.maxUploadSize 100
```

Open `http://localhost:8501` on that same computer. CPU inference is the default.
The complete checkpoint and detector are included; the training dataset and a
pretrained-weight download are unnecessary for inference. A GPU is optional.
The file watcher is explicitly disabled to avoid the prior torch.classes warning.
Inference decodes the video using OpenCV; FFprobe is not used by this new runtime.

This standalone app is intentionally separate from the existing project's
`visual.json` artifact format. Copying `model.pt` over `visual.json` will not work.
No trained checkpoint is supplied with this notebook: training must finish first.

## Expected limits

Eight sampled frames are a resource-conscious starting point, not a guarantee of
capturing brief manipulations. Long videos over 120 seconds and videos with fewer
than three usable samples are rejected. Frame-local artifacts are modeled; this
architecture does not explicitly learn motion or temporal consistency. Unknown
sources, heavy recompression, side profiles and unusual manipulation methods may
still produce uncertain or wrong answers. More GPU time alone does not fix biased
data. No accuracy or training-time target can be promised before running evaluation.
