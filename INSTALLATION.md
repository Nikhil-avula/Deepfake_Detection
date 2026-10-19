# Installation and startup

Use Python 3.11 or 3.12 and FFmpeg/FFprobe on PATH. Current verification used Python
3.12.14, Linux, CPU PyTorch 2.6.0, approximately four CPU cores of container quota and
32 GiB memory limit. The model runs on CPU. The expanded candidate dataset contains
638 balanced files; raw data, feature caches and dependencies require additional storage.

## Prepared cloud machine

```bash
cd /workspace/Deepfake_Detection
bash scripts/setup.sh
bash scripts/validate.sh
bash scripts/run.sh
```

Scripts use `/workspace/deepguard-venv` and writable runtime caches. Each cloud task is
already isolated; use the existing checkout rather than creating another Git worktree.

## Fresh Linux computer

Install FFmpeg through the distribution package manager if necessary. From the checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip check
pytest -q
streamlit run app/main.py --server.maxUploadSize 100 --browser.gatherUsageStats false
```

## Windows PowerShell

Install Python 3.12 and FFmpeg from official sources and put its bin directory on PATH:

One option is `winget install --id Gyan.FFmpeg --exact`. After installation, **close and
reopen PowerShell (and VS Code if used)** so the new PATH is inherited. Verify:

```powershell
ffmpeg -version
ffprobe -version
Test-Path .\checkpoints\visual.json
```

Run from the repository root. If either executable is not recognized, add the folder
containing `ffmpeg.exe` and `ffprobe.exe` to your user PATH. Restart already running
terminals and the application afterward.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip check
pytest -q
streamlit run app/main.py --server.maxUploadSize 100 --browser.gatherUsageStats false
```

Windows instructions have not been executed on a Windows machine. Tests needing the
owner's local media/model skip in fresh clones; portable engineering tests still run.
Raw owner videos and private caches are not published. The small trained head
`checkpoints/visual.json` and the additional-search classifier
`checkpoints/coverage_retry.json` are included in Git. Use `git pull` to get it; no retraining is
needed just to run inference. The official ImageNet backbone downloads through torchvision
on first inference, with integrity/TLS verification intact.
The generic ImageNet initialization is not itself a trained deepfake detector.
Keep the assets/ folder with the checkout: it contains the checksummed OpenCV SSD
face-detector weights and network definition. No separate face-detector download is
needed. The model and configs/default.yaml must come from the same training version.

`[WinError 2]` before any face is checked indicates an executable/setup problem, not a
lack of training data. The application now shows Setup required for missing FFprobe or
the trained head instead of implying that the video has unsuitable faces. This behavior
is regression-tested; Windows itself is not available in this execution environment.

## Repair missing OpenCV functions on Windows

An error such as `module 'cv2' has no attribute 'CascadeClassifier'` indicates an
incomplete, conflicting, or shadowed OpenCV installation. It is unrelated to training data.
Stop Streamlit, open PowerShell in the updated project root, activate its virtual
environment, and use the same Python interpreter for installation and startup:

```powershell
python -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless
python -m pip install --no-cache-dir opencv-contrib-python==4.11.0.86
python -c "import cv2; print(cv2.__file__); assert hasattr(cv2, 'CascadeClassifier'); assert hasattr(cv2.dnn, 'readNetFromCaffe'); print('OpenCV ready')"
python -m pip install -e .
python -m streamlit run app/main.py --server.fileWatcherType none
```

If verification still fails, check that the project has no local `cv2.py` file or
`cv2` directory shadowing the installed package. Only one OpenCV distribution
should be installed in this environment. Extract the entire updated review archive
into a fresh folder so code, configuration, assets and the trained model match.
The included Streamlit configuration disables its development source watcher to
avoid the separate `torch.classes` inspection warning. Restart manually after edits.

## Start Windows with the correct environment

After completing the virtual-environment installation above, use:

```powershell
.\run_windows.cmd
```

This uses this checkout's `.venv` and source directory explicitly, verifies FFprobe,
face assets, both trained classifiers and the image encoder, and disables the
Streamlit/PyTorch file-watcher conflict. It stops with a named failed check if setup
is incomplete. The encoder may download its verified official weights on first use.
For a printable diagnostic independently of the launcher:

```powershell
.\.venv\Scripts\python.exe -m deepguard.cli doctor --encoder
```

A successful startup check establishes environment readiness, not classification
accuracy. Known model errors remain listed in MODEL_RELIABILITY_AUDIT.md.
