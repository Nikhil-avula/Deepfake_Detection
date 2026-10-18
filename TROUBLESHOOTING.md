# Troubleshooting the video checker

| Message/problem | What to do |
|---|---|
| Setup required / WinError 2 | Install FFmpeg including ffprobe.exe, add its bin folder to PATH, then reopen the terminal and restart Streamlit |
| Trained model file missing | Run git pull from the repository root; checkpoints/visual.json is now included, while raw owner videos remain private |
| Cannot analyze | Read the specific reason; use a short, clear, single-person video |
| Too few faces | Try a front-facing subject, better light or higher resolution |
| Several faces | Use a clip with one main person; ambiguous samples are skipped |
| Blurry face | Use a clearer original; do not interpret rejection as a fake verdict |
| Not sure | Obtain the original source or independent evidence; do not force a label |
| Wrong answer on an unfamiliar video | Record the error and label provenance; the current pilot has limited generalization |
| Training example warning | Use held-out sources for evaluation; a familiar clip is not an independent test |
| Model not trained | Supply the actual local head artifact or train with authorized data |
| FFmpeg/FFprobe missing | Install the official distribution/package and add both executables to PATH |
| Backbone integrity error | Replace the cache through torchvision's verified official download; never bypass checks |
| Leakage error | Repair grouping/labels; do not disable the safeguard |
| Missing dataset path | Raw videos/manifests are intentionally not distributed with Git |
| Port 8501 busy | Check whether the project server is already running; stop only your own server |
| Read-only cache warning | Use the writable cache exports in scripts/setup.sh or scripts/run.sh |

Retain exact commands and failures. Do not invent results, disable assertions or make a
real video fake just to satisfy a demonstration. Use the saved per-file predictions and
exclusion records to explain failures.
