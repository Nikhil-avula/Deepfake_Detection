# Installation and startup

Use Python 3.11 or 3.12 and FFmpeg/FFprobe on PATH. Current verification used Python
3.12.14, Linux, CPU PyTorch 2.6.0, approximately four CPU cores of container quota and
32 GiB memory limit. The model runs on CPU. Raw data in this pilot total about 147 MiB;
feature caches and dependencies require additional storage.

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
`checkpoints/visual.json` is included in Git. Use `git pull` to get it; no retraining is
needed just to run inference. The official ImageNet backbone downloads through torchvision
on first inference, with integrity/TLS verification intact.
The generic ImageNet initialization is not itself a trained deepfake detector.

`[WinError 2]` before any face is checked indicates an executable/setup problem, not a
lack of training data. The application now shows Setup required for missing FFprobe or
the trained head instead of implying that the video has unsuitable faces. This behavior
is regression-tested; Windows itself is not available in this execution environment.
