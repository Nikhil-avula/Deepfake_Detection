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
No raw owner videos or trained artifacts are published by default. Install official
EfficientNet initialization through torchvision with integrity/TLS verification intact.
The generic ImageNet initialization is not itself a trained deepfake detector.
