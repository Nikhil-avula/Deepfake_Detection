# Installation

Python 3.11 or 3.12, FFmpeg/FFprobe on PATH, and enough storage for authorized datasets
are required. CPU inference works. A CUDA GPU is recommended for actual EfficientNet
and synchronization training; choose CUDA PyTorch wheels using its official selector
rather than assuming this CPU environment has a GPU.

## Linux, CPU

Install FFmpeg using your distribution package manager if absent (`sudo apt-get install
ffmpeg` on Debian/Ubuntu). Then from this checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e '.[landmarks]' --no-deps
python -m pip install mediapipe==0.10.32
python -m pip check
ffmpeg -version
deepguard hardware
pytest -q
streamlit run app/main.py --server.headless true --server.maxUploadSize 100 --browser.gatherUsageStats false
```

The landmark library alone does not include the required task model. See DATASET_GUIDE.
Torch CPU wheels must be installed **before** requirements to avoid downloading CUDA
runtime packages unnecessarily. XGBoost's Linux wheel independently pulls an NCCL
dependency; this does not establish GPU availability.

## Windows PowerShell

Install Python 3.12 and FFmpeg from its official distribution links
<https://ffmpeg.org/download.html>; add the FFmpeg `bin` directory to PATH.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e '.[landmarks]'
python -m pip check
ffprobe -version
deepguard hardware
pytest -q
streamlit run app/main.py --server.maxUploadSize 100 --browser.gatherUsageStats false
```

Windows commands are provided for reproduction but have not been executed on Windows.
Do not disable TLS, wheel verification or package-signature checks to fix an install.
Pinned direct dependencies are in requirements.txt; the actual complete resolved CPU
environment is captured in experiments/environment.freeze.txt after installation.
For a fresh platform, resolve its appropriate wheels and record a new freeze.

The current machine uses `/workspace/deepguard-venv`, outside the repository. Use this
existing checkout; cloud tasks are already isolated and need no extra Git worktree.
