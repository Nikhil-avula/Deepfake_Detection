"""Bounded media decoding, face tracking and timestamp-aligned audio features."""
from __future__ import annotations
import json
import subprocess
import tempfile
from pathlib import Path
import cv2
import librosa
import numpy as np
from .utils import sha256


def probe(path: Path) -> dict:
    try:
        result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                                 "-of", "json", str(path)], capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise ValueError("FFprobe cannot decode media")
        info = json.loads(result.stdout)
        video = next(s for s in info["streams"] if s["codec_type"] == "video")
        audio = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
        duration = float(info["format"].get("duration", video.get("duration", 0)))
        num, den = map(float, video.get("avg_frame_rate", "0/1").split("/"))
        fps = num / den if den else 0
        if not 0 < duration or not 0 < fps or int(video["width"]) * int(video["height"]) > 3840 * 2160:
            raise ValueError("Invalid duration/FPS or dimensions exceed 4K limit")
        return {"duration": duration, "fps": fps, "width": int(video["width"]),
                "height": int(video["height"]), "codec": video["codec_name"],
                "has_audio": audio is not None,
                "audio_offset": float(audio.get("start_time", 0)) - float(video.get("start_time", 0)) if audio else 0}
    except (StopIteration, KeyError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Invalid or unsupported video") from exc


def extract_audio(path: Path, sr: int, duration: float) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="deepguard-audio-") as folder:
        output = Path(folder) / "audio.wav"
        result = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(path),
                                 "-vn", "-t", str(duration), "-ac", "1", "-ar", str(sr),
                                 "-acodec", "pcm_s16le", str(output)], capture_output=True, timeout=90)
        if result.returncode:
            raise ValueError("Audio extraction failed")
        return librosa.load(output, sr=sr, mono=True)[0]


def mel(audio: np.ndarray, sr: int, n_mels: int = 64) -> np.ndarray:
    power = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=400, hop_length=160, n_mels=n_mels)
    db = librosa.power_to_db(power, ref=1.0, top_db=80)
    # Fixed dB range, not statistics fitted on validation/test data.
    return np.clip((db + 80) / 80, 0, 1).astype(np.float32)


def audio_window(audio: np.ndarray, center: float, seconds: float, sr: int) -> np.ndarray | None:
    start = round((center - seconds / 2) * sr)
    count = round(seconds * sr)
    if start < 0 or start + count > len(audio):
        return None
    return audio[start:start + count]


def iou(a: tuple, b: tuple) -> float:
    x, y = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    intersection = max(0, right - x) * max(0, bottom - y)
    return intersection / max(1, a[2] * a[3] + b[2] * b[3] - intersection)


class MouthLandmarks:
    """Optional MediaPipe FaceLandmarker asset; no heuristic masquerades as landmarks."""
    def __init__(self, asset: str):
        import mediapipe as mp
        self.mp = mp
        self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=asset), num_faces=1))

    def crop(self, rgb: np.ndarray) -> np.ndarray | None:
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self.detector.detect(image)
        if not result.face_landmarks:
            return None
        points = result.face_landmarks[0]
        ids = [61, 291, 0, 17, 78, 308, 13, 14]
        h, w = rgb.shape[:2]
        xs, ys = [points[i].x * w for i in ids], [points[i].y * h for i in ids]
        pad = max(4, (max(xs) - min(xs)) * .2)
        x1, x2 = max(0, int(min(xs) - pad)), min(w, int(max(xs) + pad))
        y1, y2 = max(0, int(min(ys) - pad)), min(h, int(max(ys) + pad))
        if x2 <= x1 or y2 <= y1:
            return None
        return cv2.resize(rgb[y1:y2, x1:x2], (64, 64))

    def close(self) -> None:
        self.detector.close()


def preprocess(path: str | Path, cfg: dict, cache: str | Path | None = None,
               landmark_asset: str | None = None) -> dict:
    path = Path(path)
    key = sha256(path) + "-" + __import__("hashlib").sha256(
        json.dumps({"cfg": cfg, "asset": sha256(landmark_asset) if landmark_asset else None,
                    "schema": 2}, sort_keys=True).encode()).hexdigest()[:16]
    cached = Path(cache) / f"{key}.npz" if cache else None
    if cached and cached.exists():
        with np.load(cached, allow_pickle=False) as values:
            return {k: values[k] for k in values.files}
    info = probe(path)
    if path.stat().st_size > cfg["max_upload_mb"] * 1024 ** 2 or info["duration"] > cfg["max_duration"]:
        raise ValueError("Video exceeds configured size or duration limit")
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    landmarks = MouthLandmarks(landmark_asset) if landmark_asset else None
    capture = cv2.VideoCapture(str(path))
    faces, times, mouths, mouth_times, quality, selected = [], [], [], [], [], []
    previous = None
    sampled = multiple = misses = 0
    try:
        next_time = 0.0
        while sampled < cfg["max_frames"]:
            ok, bgr = capture.read()
            if not ok:
                break
            t = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if t + 1e-4 < next_time:
                continue
            next_time = t + 1 / cfg["sample_fps"]
            sampled += 1
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            boxes = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            if len(boxes) != 1:
                multiple += int(len(boxes) > 1)
                misses += int(len(boxes) == 0)
                continue
            box = tuple(map(int, boxes[0]))
            if previous is not None and iou(previous, box) < .1:
                misses += 1
                continue
            previous = box
            x, y, w, h = box
            rgb = cv2.cvtColor(bgr[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
            face = cv2.resize(rgb, (cfg["image_size"], cfg["image_size"]))
            faces.append(face)
            times.append(t)
            sharpness = cv2.Laplacian(gray[y:y+h, x:x+w], cv2.CV_64F).var()
            quality.append(min(1., min(w, h) / 160) * min(1., sharpness / 100))
            if len(selected) < 6:
                selected.append(cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (320, 180)))
            mouth = landmarks.crop(rgb) if landmarks else None
            if mouth is not None:
                mouths.append(mouth)
                mouth_times.append(t)
    finally:
        capture.release()
        if landmarks:
            landmarks.close()
    sr = cfg["sample_rate"]
    audio = extract_audio(path, sr, info["duration"]) if info["has_audio"] else np.empty(0, dtype=np.float32)
    rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.
    clipped = float(np.mean(np.abs(audio) >= .999)) if len(audio) else 0.
    segments = []
    count = sr * cfg["audio_seconds"]
    # Full windows only; silence is not called speech. RMS is only a usability gate.
    for start in range(0, max(0, len(audio) - count + 1), count):
        segment = audio[start:start+count]
        if np.sqrt(np.mean(segment ** 2)) > .005:
            segments.append(mel(segment, sr, cfg["n_mels"]))
    sync_mouths, sync_mels, sync_times = [], [], []
    for mouth, t in zip(mouths, mouth_times):
        segment = audio_window(audio, t - info["audio_offset"], .2, sr)
        if segment is not None and np.sqrt(np.mean(segment ** 2)) > .005:
            sync_mouths.append(mouth)
            sync_mels.append(mel(segment, sr, cfg["n_mels"]))
            sync_times.append(t)
    fraction = len(faces) / max(1, sampled)
    result = {"faces": np.asarray(faces, dtype=np.uint8).reshape(-1, cfg["image_size"], cfg["image_size"], 3),
              "times": np.asarray(times), "selected_frames": np.asarray(selected, dtype=np.uint8),
              "audio_mels": np.asarray(segments, dtype=np.float32),
              "mouths": np.asarray(sync_mouths, dtype=np.uint8).reshape(-1, 64, 64, 3),
              "sync_mels": np.asarray(sync_mels, dtype=np.float32), "sync_times": np.asarray(sync_times),
              "visual_quality": np.asarray(float(np.mean(quality)) * fraction if quality else 0.),
              "audio_quality": np.asarray(float(bool(segments)) * max(0., 1 - clipped * 10)),
              "face_fraction": np.asarray(fraction), "multiple_faces": np.asarray(multiple),
              "missing_faces": np.asarray(misses), "sampled_frames": np.asarray(sampled),
              "audio_rms": np.asarray(rms), "audio_clipped": np.asarray(clipped),
              "metadata": np.asarray(json.dumps(info)), "schema": np.asarray(2)}
    if cached:
        cached.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cached, **result)
    return result


def handcrafted(features: dict) -> np.ndarray:
    """Texture, spectral, temporal and availability statistics for classic baselines."""
    faces = features["faces"]
    visual = []
    for face in faces:
        gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        visual.append([gray.mean()/255, gray.std()/255,
                       np.log1p(cv2.Laplacian(gray, cv2.CV_64F).var()),
                       (cv2.Canny(gray, 80, 160) > 0).mean()])
    visual = np.asarray(visual)
    v = np.r_[visual.mean(0), visual.std(0)] if len(visual) else np.zeros(8)
    motion = float(np.abs(np.diff(faces.astype(float), axis=0)).mean()/255) if len(faces)>1 else 0.
    audio = features["audio_mels"]
    a = [float(audio.mean()), float(audio.std())] if audio.size else [0., 0.]
    return np.asarray([*v, motion, *a, float(features["visual_quality"]),
                       float(features["audio_quality"]), float(features["face_fraction"])], dtype=np.float32)
