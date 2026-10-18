"""Video-only decoding, whole-clip sampling, conservative face crops and quality gates."""
from __future__ import annotations
import json
import math
import subprocess
from pathlib import Path
import cv2
import numpy as np


class MediaError(ValueError):
    """An upload cannot safely provide useful facial evidence."""


def probe(path: str | Path, cfg: dict) -> dict:
    path=Path(path)
    if not path.is_file() or not 0 < path.stat().st_size <= cfg["max_upload_mb"]*1024**2:
        raise MediaError("The file is empty, missing or larger than the upload limit.")
    try:
        p=subprocess.run(["ffprobe","-v","error","-show_streams","-show_format","-of","json",str(path)],capture_output=True,text=True,timeout=20)
        if p.returncode:raise MediaError("This file could not be read as a video. Try an MP4 file.")
        data=json.loads(p.stdout)
        v=next(x for x in data["streams"] if x["codec_type"]=="video")
        duration=float(data["format"].get("duration",v.get("duration",0)))
        n,d=map(float,v.get("avg_frame_rate","0/1").split("/")); fps=n/d if d else 0
        w,h=int(v["width"]),int(v["height"])
        if not math.isfinite(duration+fps) or duration<=0 or duration>cfg["max_duration"]:
            raise MediaError(f"Please use a video up to {cfg['max_duration']} seconds long.")
        if fps<=0 or w<=0 or h<=0 or w*h>cfg["max_pixels"]:
            raise MediaError("Unsupported video frame rate or resolution. Use up to 4K.")
        return {"duration":duration,"fps":fps,"width":w,"height":h,"codec":v["codec_name"],"bytes":path.stat().st_size}
    except (KeyError,StopIteration,ValueError,json.JSONDecodeError,subprocess.TimeoutExpired) as exc:
        if isinstance(exc,MediaError):raise
        raise MediaError("The video is damaged or uses an unsupported format.") from exc


def sample_video(path: str | Path,cfg: dict) -> dict:
    info=probe(path,cfg)
    # Sample across the complete clip, rather than only its first seconds.
    target=np.linspace(0,max(0,info["duration"]-1/max(info["fps"],1)),cfg["frames"])
    capture=cv2.VideoCapture(str(path))
    detector=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    faces,times,sharpness,sizes,frames=[],[],[],[],[]
    missing=multiple=decoded=0
    try:
        for t in target:
            capture.set(cv2.CAP_PROP_POS_MSEC,float(t*1000))
            ok,bgr=capture.read()
            if not ok:continue
            decoded+=1
            if len(frames)<4:frames.append(cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),(320,180)))
            gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
            boxes=detector.detectMultiScale(gray,scaleFactor=1.08,minNeighbors=4,minSize=(35,35))
            if len(boxes)!=1:
                missing+=int(len(boxes)==0);multiple+=int(len(boxes)>1);continue
            x,y,w,h=map(int,boxes[0])
            # Retain a little blending boundary context, applied to every class.
            pad=round(max(w,h)*.15)
            x1,y1=max(0,x-pad),max(0,y-pad)
            crop=bgr[y1:min(bgr.shape[0],y+h+pad),x1:min(bgr.shape[1],x+w+pad)]
            rgb=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
            faces.append(cv2.resize(rgb,(cfg["image_size"],cfg["image_size"])))
            times.append(float(t));sizes.append(min(w,h))
            sharpness.append(float(cv2.Laplacian(gray[y:y+h,x:x+w],cv2.CV_64F).var()))
    finally:capture.release()
    if not decoded:raise MediaError("No video frames could be decoded.")
    reasons=[]
    if multiple/decoded>cfg["max_multiple_fraction"]:
        reasons.append("Too many frames contain several faces. Please use a single-person video.")
    if len(faces)<cfg["min_faces"] or len(faces)/decoded<cfg["min_face_fraction"]:
        reasons.append("Too few clear front-facing faces were found.")
    if sharpness and np.median(sharpness)<cfg["min_sharpness"]:
        reasons.append("The face looks too blurry for a useful result.")
    return {"metadata":info,"faces":np.asarray(faces,dtype=np.uint8).reshape(-1,cfg["image_size"],cfg["image_size"],3),
            "times":np.asarray(times),"sharpness":np.asarray(sharpness),"sizes":np.asarray(sizes),
            "frames":np.asarray(frames),"sampled":decoded,"missing":missing,"multiple":multiple,
            "usable":not reasons,"reasons":reasons}


def texture_features(faces: np.ndarray) -> np.ndarray:
    """Eight texture statistics, independent of filenames and video container labels."""
    values=[]
    for face in faces:
        gray=cv2.cvtColor(face,cv2.COLOR_RGB2GRAY)
        values.append([gray.mean()/255,gray.std()/255,np.log1p(cv2.Laplacian(gray,cv2.CV_64F).var()),
                       np.mean(cv2.Canny(gray,80,160)>0)])
    if not values:return np.zeros(8,dtype=np.float32)
    x=np.asarray(values)
    return np.r_[x.mean(0),x.std(0)].astype(np.float32)
