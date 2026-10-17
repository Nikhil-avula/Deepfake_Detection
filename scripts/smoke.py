"""Actual CLI end-to-end engineering check; never a deepfake benchmark."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

with tempfile.TemporaryDirectory(prefix="deepguard-smoke-") as folder:
    video = Path(folder)/"engineering-generated.mp4"
    report = Path(folder)/"report.json"
    subprocess.run(["ffmpeg","-nostdin","-v","error","-y","-f","lavfi","-i",
                    "color=c=blue:s=160x120:r=25:d=5","-f","lavfi","-i",
                    "sine=frequency=440:sample_rate=16000:duration=5",
                    "-c:v","libx264","-pix_fmt","yuv420p","-c:a","aac","-shortest",str(video)],check=True)
    subprocess.run([sys.executable,"-m","deepguard.cli","analyze",str(video),str(report),
                    "--checkpoints",str(Path(folder)/"no-weights")],check=True)
    result = json.loads(report.read_text())
    assert result["decision"] == "Insufficient Evidence"
    assert not result["probabilities"]
    assert "Model not trained" in result["reasons"]
    assert report.with_suffix(".html").exists()
    print(json.dumps({"test":"Generated color/sine fixture, CLI and HTML report",
                      "outcome":"passed", "decision":result["decision"],
                      "analysis_seconds":result["seconds"],"research_metrics":None},indent=2))
