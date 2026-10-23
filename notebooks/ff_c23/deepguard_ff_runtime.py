"""Shared face preprocessing and inference for Kaggle and local computers."""
from pathlib import Path
import argparse
import hashlib
import json
import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

ASSETS = {
    'face_res10.prototxt': 'dcd661dc48fc9de0a341db1f666a2164ea63a67265c7f779bc12d6b3f2fa67e9',
    'face_res10.caffemodel': '510ffd2471bd81e3fcc88a5beb4eae4fb445ccf8333ebc54e7302b83f4158a76',
}
PREPROCESS = dict(frames=8, image_size=224, confidence=0.8, margin=0.2,
                  dominant_area_ratio=4.0, min_faces=3, max_duration=120.0)


def verify_assets(folder):
    folder = Path(folder)
    for name, digest in ASSETS.items():
        path = folder / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Missing or damaged face detector: {path}')


def build_model(pretrained=False):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model


def face_tensor(face):
    image = Image.fromarray(face).resize((224, 224), Image.Resampling.BILINEAR)
    x = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float().div(255)
    mean = x.new_tensor([.485, .456, .406])[:, None, None]
    std = x.new_tensor([.229, .224, .225])[:, None, None]
    return (x - mean) / std


def extract_faces(path, detector, cfg):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError('Video could not be opened')
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if count < 1 or not np.isfinite(fps) or fps <= 0:
            raise ValueError('Video has invalid frame count or frame rate')
        if count / fps > cfg['max_duration']:
            raise ValueError(f"Video exceeds {cfg['max_duration']} seconds; trim it first")
        positions = np.unique(np.linspace(0, count - 1, cfg['frames'] + 2).astype(int)[1:-1])
        faces = []
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
            ok, bgr = cap.read()
            if not ok:
                continue
            h, w = bgr.shape[:2]
            detector.setInput(cv2.dnn.blobFromImage(bgr, 1., (300, 300), (104., 177., 123.)))
            detections = detector.forward()[0, 0]
            boxes, scores = [], []
            for d in detections:
                if float(d[2]) < cfg['confidence']:
                    continue
                x1, y1, x2, y2 = (d[3:7] * np.array([w, h, w, h])).astype(int)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                if min(x2 - x1, y2 - y1) >= 32:
                    boxes.append([int(x1), int(y1), int(x2-x1), int(y2-y1)])
                    scores.append(float(d[2]))
            if not boxes:
                continue
            keep = np.asarray(cv2.dnn.NMSBoxes(boxes, scores, cfg['confidence'], .3)).reshape(-1)
            boxes = sorted([boxes[int(i)] for i in keep], key=lambda b: b[2]*b[3], reverse=True)
            if not boxes:
                continue
            if len(boxes) > 1 and boxes[0][2]*boxes[0][3] < cfg['dominant_area_ratio']*boxes[1][2]*boxes[1][3]:
                continue
            x, y, bw, bh = boxes[0]
            mx, my = int(bw*cfg['margin']), int(bh*cfg['margin'])
            crop = bgr[max(0,y-my):min(h,y+bh+my), max(0,x-mx):min(w,x+bw+mx)]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            faces.append(np.asarray(Image.fromarray(crop).resize(
                (cfg['image_size'], cfg['image_size']), Image.Resampling.BILINEAR)))
        return faces, len(positions)
    finally:
        cap.release()


def predict_video(video, bundle, device='cpu'):
    bundle = Path(bundle)
    cfg = json.loads((bundle/'config.json').read_text())
    verify_assets(bundle)
    model_path = bundle/'model.pt'
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != cfg['model_sha256']:
        raise RuntimeError('Model integrity check failed')
    model = build_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    detector = cv2.dnn.readNetFromCaffe(str(bundle/'face_res10.prototxt'), str(bundle/'face_res10.caffemodel'))
    faces, sampled = extract_faces(video, detector, cfg['preprocessing'])
    result = dict(decision='Cannot analyze', faces_found=len(faces), frames_sampled=sampled,
                  fake_score=None, notice='Research model; the score is not proof of authenticity.')
    if len(faces) < cfg['preprocessing']['min_faces']:
        result['reason'] = 'Not enough usable face samples; use a clear, short video with one foreground person.'
        return result
    # Cache training crops use JPEG quality=95; reproduce that step for local inference.
    import io
    decoded = []
    for face in faces:
        stream = io.BytesIO()
        Image.fromarray(face).save(stream, format='JPEG', quality=95)
        stream.seek(0)
        decoded.append(np.asarray(Image.open(stream).convert('RGB')))
    with torch.inference_mode():
        logits = model(torch.stack([face_tensor(f) for f in decoded]).to(device)).flatten()
        score = float(logits.mean().sigmoid().cpu())
    low, high = cfg['real_threshold'], cfg['fake_threshold']
    result.update(fake_score=score, decision='Likely real' if score <= low else 'Likely fake' if score >= high else 'Not sure')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video')
    parser.add_argument('--bundle', default=str(Path(__file__).resolve().parent))
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    print(json.dumps(predict_video(args.video, args.bundle, args.device), indent=2))
