"""Video-only decoding, whole-clip sampling, conservative face crops and quality gates."""
from __future__ import annotations
import json
import math
import subprocess
import shutil
from pathlib import Path
import cv2
import numpy as np
from functools import lru_cache
from .utils import ROOT,sha256


class MediaError(ValueError):
    """An upload cannot safely provide useful facial evidence."""


class SetupError(RuntimeError):
    """A machine dependency is missing; the uploaded video is not at fault."""


@lru_cache(maxsize=1)
def neural_face_assets():
    model=ROOT/'assets/face_res10.caffemodel';definition=ROOT/'assets/face_res10.prototxt'
    for path,expected in [(model,'510ffd2471bd81e3fcc88a5beb4eae4fb445ccf8333ebc54e7302b83f4158a76'),
                          (definition,'dcd661dc48fc9de0a341db1f666a2164ea63a67265c7f779bc12d6b3f2fa67e9')]:
        if not path.is_file() or sha256(path)!=expected:
            raise SetupError('The bundled face detector is missing or damaged. Restore the assets folder from the project checkout.')
    return str(definition),str(model)


def detect_boxes(detector,bgr,kind):
    if kind=='haar':
        return detector.detectMultiScale(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),scaleFactor=1.08,minNeighbors=4,minSize=(35,35))
    h,w=bgr.shape[:2]
    detector.setInput(cv2.dnn.blobFromImage(bgr,1.,(300,300),(104.,177.,123.),swapRB=False,crop=False))
    output=detector.forward()[0,0];boxes=[];scores=[]
    for item in output:
        if item[2]<.9:continue
        x1,y1,x2,y2=item[3:7]*np.asarray([w,h,w,h])
        x1,y1=max(0,int(x1)),max(0,int(y1));x2,y2=min(w,int(x2)),min(h,int(y2))
        if min(x2-x1,y2-y1)<20:continue
        boxes.append([x1,y1,x2-x1,y2-y1]);scores.append(float(item[2]))
    keep=cv2.dnn.NMSBoxes(boxes,scores,.9,.3)
    return [boxes[int(i)] for i in np.asarray(keep).reshape(-1)]


def primary_face(boxes,area_ratio: float):
    """Keep a clearly dominant foreground face; abstain when people are comparable in size."""
    if len(boxes)==1:return boxes[0]
    if len(boxes)>1:
        ordered=sorted(boxes,key=lambda b:int(b[2])*int(b[3]),reverse=True)
        if ordered[0][2]*ordered[0][3]>=area_ratio*ordered[1][2]*ordered[1][3]:return ordered[0]
    return None


def sampled_frames(capture,target,fps,mode):
    """Decode once in order, avoiding repeated H264 keyframe seeks."""
    if mode not in ('seek','sequential_v1'):raise SetupError('Unknown video sampling configuration.')
    position=-1
    for t in target:
        if mode=='seek':
            capture.set(cv2.CAP_PROP_POS_MSEC,float(t*1000))
            ok,bgr=capture.read()
        else:
            desired=int(round(float(t)*fps))
            while position<desired:
                if not capture.grab():return
                position+=1
            ok,bgr=capture.retrieve()
        if ok:yield float(t),bgr


def probe(path: str | Path, cfg: dict) -> dict:
    path=Path(path)
    if not path.is_file() or not 0 < path.stat().st_size <= cfg["max_upload_mb"]*1024**2:
        raise MediaError("The file is empty, missing or larger than the upload limit.")
    if not shutil.which("ffprobe"):
        raise SetupError("FFprobe is not installed or is not on PATH. Install FFmpeg including ffprobe, restart your terminal and restart the application. See INSTALLATION.md.")
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
    except FileNotFoundError as exc:
        raise SetupError("Windows could not start ffprobe. Check that ffprobe.exe is installed and its bin folder is on PATH, then restart the application.") from exc
    except (KeyError,StopIteration,ValueError,json.JSONDecodeError,subprocess.TimeoutExpired) as exc:
        if isinstance(exc,MediaError):raise
        raise MediaError("The video is damaged or uses an unsupported format.") from exc


def sample_video(path: str | Path,cfg: dict) -> dict:
    info=probe(path,cfg)
    # Sample across the complete clip, rather than only its first seconds.
    target=np.linspace(0,max(0,info["duration"]-1/max(info["fps"],1)),cfg["frames"])
    kind=cfg.get('face_detector','haar')
    if kind not in ('haar','res10'):raise SetupError('Unknown face detector configuration.')
    # Initialize before opening the video so failed setup cannot leave a Windows file lock.
    try:
        detector=cv2.dnn.readNetFromCaffe(*neural_face_assets()) if kind=='res10' else cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    except (AttributeError,cv2.error) as exc:
        raise SetupError('OpenCV could not initialize the face detector. Your OpenCV installation may be incomplete or conflicting. Follow the OpenCV repair steps in INSTALLATION.md and restart the app.') from exc
    faces,times,sharpness,sizes,frames=[],[],[],[],[]
    preview_times=target[np.rint(np.linspace(0,len(target)-1,min(4,len(target)))).astype(int)]
    missing=multiple=decoded=dominant=0
    capture=cv2.VideoCapture(str(path))
    try:
        for t,bgr in sampled_frames(capture,target,info['fps'],cfg.get('video_sampler','seek')):
            decoded+=1
            if np.any(np.isclose(t,preview_times)):frames.append(cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),(320,180)))
            gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
            boxes=detect_boxes(detector,bgr,kind)
            box=primary_face(boxes,cfg.get('dominant_face_area_ratio',float('inf')))
            if box is None:
                missing+=int(len(boxes)==0);multiple+=int(len(boxes)>1);continue
            dominant+=int(len(boxes)>1)
            x,y,w,h=map(int,box)
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
        reasons.append("Several similarly sized faces appear. Use a video with one main person, or crop to the person you want to check.")
    if len(faces)<cfg["min_faces"] or len(faces)/decoded<cfg["min_face_fraction"]:
        reasons.append(f"Found {len(faces)} usable face samples in {decoded} frames; at least {cfg['min_faces']} samples and {cfg['min_face_fraction']:.0%} coverage are needed. Try a longer section where the face is visible, or crop closer to the person.")
    if sharpness and np.median(sharpness)<cfg["min_sharpness"]:
        reasons.append("The face looks too blurry for a useful result.")
    return {"metadata":info,"faces":np.asarray(faces,dtype=np.uint8).reshape(-1,cfg["image_size"],cfg["image_size"],3),
            "times":np.asarray(times),"sharpness":np.asarray(sharpness),"sizes":np.asarray(sizes),
            "frames":np.asarray(frames),"sampled":decoded,"missing":missing,"multiple":multiple,"dominant_face_frames":dominant,
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
