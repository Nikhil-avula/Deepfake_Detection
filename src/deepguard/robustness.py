"""Fixed held-out visual degradations; source relationships and labels are retained."""
from pathlib import Path
import subprocess


def perturb(source: str,output: str,kind: str) -> None:
    args={"compression":["-c:v","libx264","-crf","35"],
          "blur":["-vf","gblur=sigma=1.2","-c:v","libx264","-crf","18"],
          "darkness":["-vf","eq=brightness=-0.12","-c:v","libx264","-crf","18"]}
    if kind not in args:raise ValueError("Unknown visual degradation")
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    p=subprocess.run(["ffmpeg","-nostdin","-v","error","-y","-i",source,"-an",*args[kind],output],capture_output=True,timeout=120)
    if p.returncode:raise ValueError("Could not produce the visual robustness input")
