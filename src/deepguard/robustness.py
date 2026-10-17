"""Deterministic FFmpeg perturbations for paired robustness experiments."""
from pathlib import Path
import subprocess


def perturb(source: str, output: str, kind: str) -> None:
    filters = {"compression": ["-c:v", "libx264", "-crf", "38"],
               "blur": ["-vf", "gblur=sigma=1.5", "-c:v", "libx264", "-crf", "18"],
               "lighting": ["-vf", "eq=brightness=-0.12", "-c:v", "libx264", "-crf", "18"],
               "noise": ["-af", "aecho=0.8:0.9:40:0.2", "-c:v", "copy"]}
    if kind not in filters:
        raise ValueError("Unknown perturbation")
    if kind == "noise":
        # Seeded additive white noise at fixed amplitude; record this severity.
        options = ["-filter_complex", "[0:a]asplit=2[a][b];[b]volume=0,aeval=0.02*(2*random(42)-1)[n];[a][n]amix=inputs=2:normalize=0[out]", "-map", "0:v", "-map", "[out]", "-c:v", "copy"]
    else:
        options = filters[kind] + ["-c:a", "aac"]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", source,
                             *options, output], capture_output=True, timeout=180)
    if result.returncode:
        raise ValueError("Perturbation failed: " + result.stderr.decode(errors="replace")[-500:])
