# %% [markdown]
# DeepGuard: FaceForensics++ C23 on Kaggle
# Attach `xdxd003/ff-c23` using Add Input. Enable GPU and Internet.
# This notebook never downloads the video dataset. It downloads small detector files,
# official split metadata, and ImageNet weights. Run cells in order.
# The test set is opened only in the final evaluation cell. Do not tune after seeing it.
# %%
from pathlib import Path
import os, sys, json, random, hashlib, shutil, time, re, urllib.request
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torchvision
import cv2
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score, accuracy_score,
                             confusion_matrix, classification_report)

assert torch.cuda.is_available(), 'Enable GPU in Kaggle notebook settings before training.'
print('Python:', sys.version, 'Torch:', torch.__version__, 'GPU:', torch.cuda.get_device_name(0))
print('GPU memory GiB:', round(torch.cuda.get_device_properties(0).total_memory/2**30, 1))
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(2)
cv2.setNumThreads(1)
WORK = Path('/kaggle/working/deepguard_ff')
WORK.mkdir(parents=True, exist_ok=True)
# If auto-detection finds multiple versions, set DATA_ROOT to the desired attached folder.
DATA_ROOT = Path('/kaggle/input')
# Optional: attach official split JSONs and set this folder for offline split loading.
SPLITS_DIR = None
BATCH_VIDEOS = 4          # Four videos x up to four crops = up to 16 training images.
TRAIN_FRAMES = 4
WORKERS = 2              # Set to 0 if multiprocessing fails in your notebook.
HEAD_EPOCHS = 3
FINETUNE_EPOCHS = 10
PATIENCE = 3
DEVICE = 'cuda'
print('Writable free GiB:', round(shutil.disk_usage(WORK).free/2**30, 1))
# %% [markdown]
# Shared runtime is embedded in the downloadable .ipynb cell below.
# It defines the exact same model, face cropping, JPEG decoding and normalization
# used for training and for the exported local predictor.
# %% RUNTIME_PLACEHOLDER
# Replaced by the notebook builder. In the .ipynb this cell writes and imports the runtime.
# %% [markdown]
# Verify attached dataset and labels. DeepFakeDetection is excluded from this first
# experiment because its original controls and source metadata need separate mapping.
# Unknown folder names/filenames are reported, never guessed.
# %%
METHODS = {'deepfakes', 'face2face', 'faceshifter', 'faceswap', 'neuraltextures'}
EXTS = {'.mp4', '.avi', '.mov', '.mkv'}
rows, ignored = [], []
for path in sorted(DATA_ROOT.rglob('*')):
    if not path.is_file() or path.suffix.lower() not in EXTS:
        continue
    parts = {p.casefold() for p in path.relative_to(DATA_ROOT).parts[:-1]}
    categories = parts & (METHODS | {'original', 'deepfakedetection'})
    if categories == {'deepfakedetection'}:
        ignored.append((str(path), 'DeepFakeDetection reserved, not used'))
        continue
    if len(categories) != 1:
        ignored.append((str(path), 'Unrecognized or ambiguous category')); continue
    method = next(iter(categories))
    real = method == 'original'
    match = re.fullmatch(r'(\d{3})' if real else r'(\d{3})_(\d{3})', path.stem)
    if not match:
        ignored.append((str(path), 'Unrecognized FF++ source IDs')); continue
    source_ids = list(match.groups())
    uid = f'{method}/{path.stem}'
    rows.append(dict(uid=uid, path=str(path), label=0 if real else 1,
                     method=method, source_ids=source_ids))
manifest = pd.DataFrame(rows)
assert len(manifest), 'No supported videos found. Inspect attached folders; do not guess labels.'
assert not manifest.uid.duplicated().any(), 'Duplicate dataset versions/IDs found. Narrow DATA_ROOT.'
assert set(manifest.method) == METHODS | {'original'}, 'Some expected folders are missing; inspect input.'
print(manifest.groupby(['method', 'label']).size().to_string())
print('Supported videos:', len(manifest), 'Ignored videos:', len(ignored))
pd.DataFrame(ignored, columns=['path','reason']).to_csv(WORK/'ignored_files.csv', index=False)
print('Sample paths:', manifest.path.head().tolist())
# %% [markdown]
# Load official split metadata (only a few KB). Both IDs of a manipulated video
# must belong to the same partition. Cross-partition pairs are excluded.
# IDs represent original source videos, not verified unique human identities.
# %%
split_ids = {}
for split, filename in [('train','train.json'), ('val','val.json'), ('test','test.json')]:
    if SPLITS_DIR is not None:
        pairs = json.loads((Path(SPLITS_DIR)/filename).read_text())
    else:
        url = 'https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/' + filename
        with urllib.request.urlopen(url, timeout=60) as response:
            pairs = json.load(response)
    assert isinstance(pairs, list) and pairs, f'Invalid {filename}'
    assert all(isinstance(p, list) and len(p)==2 for p in pairs), f'Unexpected split schema: {filename}'
    ids = {str(v).zfill(3) for pair in pairs for v in pair}
    assert all(re.fullmatch(r'\d{3}', v) for v in ids)
    split_ids[split] = ids
    (WORK/filename).write_text(json.dumps(pairs, indent=2))
for a,b in [('train','val'), ('train','test'), ('val','test')]:
    assert not split_ids[a] & split_ids[b], f'Official ID overlap: {a}/{b}'

def assign_split(ids):
    owners = [s for s, values in split_ids.items() if set(ids) <= values]
    return owners[0] if len(owners)==1 else 'excluded'
manifest['split'] = manifest.source_ids.map(assign_split)
manifest.to_csv(WORK/'all_manifest.csv', index=False)
print(pd.crosstab(manifest.split, manifest.method))
print('Excluded cross-partition/unmapped videos:', int((manifest.split=='excluded').sum()))
manifest = manifest[manifest.split!='excluded'].copy().reset_index(drop=True)
for split in ['train','val','test']:
    assert set(manifest.loc[manifest.split==split, 'label']) == {0,1}, f'{split} needs both labels'
# Verify grouping again using actual retained files.
actual_ids = {s: set().union(*map(set, manifest.loc[manifest.split==s,'source_ids'])) for s in split_ids}
assert not (actual_ids['train'] & actual_ids['val'] or actual_ids['train'] & actual_ids['test'] or actual_ids['val'] & actual_ids['test'])
print('Source video IDs are disjoint across partitions. Person identity overlap remains unverified.')
# %% [markdown]
# Download and verify bundled detector assets. No dataset download occurs here.
# First run downloads ImageNet weights later (about 20 MB). Detector is about 5 MB.
# %%
ASSET_DIR = WORK/'assets'; ASSET_DIR.mkdir(exist_ok=True)
ASSET_REV = 'bff43a85ddf4abe192e20f40f44d97fd7e91368d'
for name, expected in runtime.ASSETS.items():
    target = ASSET_DIR/name
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
        url = f'https://raw.githubusercontent.com/Nikhil-avula/Deepfake_Detection/{ASSET_REV}/assets/{name}'
        urllib.request.urlretrieve(url, target)
# Preserve the detector's upstream license and documentation with the exported bundle.
for name in ['OPENCV_LICENSE','RES10_UPSTREAM_README.md']:
    target=ASSET_DIR/name
    if not target.exists():
        urllib.request.urlretrieve(f'https://raw.githubusercontent.com/Nikhil-avula/Deepfake_Detection/{ASSET_REV}/assets/{name}',target)
runtime.verify_assets(ASSET_DIR)
CFG = dict(runtime.PREPROCESS)
detector = cv2.dnn.readNetFromCaffe(str(ASSET_DIR/'face_res10.prototxt'), str(ASSET_DIR/'face_res10.caffemodel'))
# %% [markdown]
# Cache up to eight face crops/video. Re-running this cell resumes finished videos.
# Stop and inspect failures if a class/method is disproportionately rejected.
# Reserve at least 3 GiB writable space for checkpoints and export.
# %%
CACHE = WORK/'faces'; CACHE.mkdir(exist_ok=True)
cache_signature = hashlib.sha256(json.dumps(CFG, sort_keys=True).encode()).hexdigest()
cache_rows = []
t0 = time.time()
for i, row in enumerate(manifest.to_dict('records')):
    folder = CACHE/row['uid']; folder.mkdir(parents=True, exist_ok=True)
    marker = folder/'done.json'
    cached = json.loads(marker.read_text()) if marker.exists() else None
    stat = Path(row['path']).stat()
    source_signature = [stat.st_size, stat.st_mtime_ns]
    if cached and cached.get('config')==cache_signature and cached.get('source')==source_signature:
        crop_paths = [str(folder/name) for name in cached['files']]
        valid = all(Path(p).exists() for p in crop_paths)
    else:
        valid = False
    if not valid:
        if shutil.disk_usage(WORK).free < 3*2**30:
            raise RuntimeError('Less than 3 GiB free. Preserve outputs and reduce cache size or use another session.')
        for old in folder.glob('*.jpg'): old.unlink()
        error = ''
        try:
            faces, sampled = runtime.extract_faces(row['path'], detector, CFG)
        except (ValueError, cv2.error) as exc:
            faces, sampled, error = [], 0, str(exc)
        names = []
        for j, face in enumerate(faces):
            name = f'{j:02d}.jpg'
            Image.fromarray(face).save(folder/name, quality=95)
            names.append(name)
        cached = dict(config=cache_signature, source=source_signature, files=names, sampled=sampled, error=error)
        marker.write_text(json.dumps(cached))
        crop_paths = [str(folder/name) for name in names]
    cache_rows.append(dict(**row, crops=crop_paths, faces=len(crop_paths),
                           usable=len(crop_paths)>=CFG['min_faces'], error=cached.get('error','')))
    if (i+1)%100==0 or i+1==len(manifest):
        print(f'{i+1}/{len(manifest)} videos; {(time.time()-t0)/60:.1f} min; free {shutil.disk_usage(WORK).free/2**30:.1f} GiB', flush=True)
prepared = pd.DataFrame(cache_rows)
prepared.drop(columns=['crops']).to_csv(WORK/'coverage.csv', index=False)
print(prepared.groupby(['split','method']).usable.agg(['count','sum','mean']).to_string())
prepared.to_json(WORK/'prepared.json', orient='records', indent=2)
usable = prepared[prepared.usable].copy()
for s in ['train','val','test']:
    assert set(usable.loc[usable.split==s,'label'])=={0,1}, f'No eligible examples in {s}'
    assert usable.loc[usable.split==s].groupby('method').size().size==6, f'Missing method in {s}'
print('Coverage excludes invalid/insufficient-face videos; accuracy must be reported alongside coverage.')
# %% [markdown]
# Inspect face crops before training. Labels shown here come from folder mapping.
# Only training examples are displayed. Do not inspect test predictions to tune.
# %%
import matplotlib.pyplot as plt
examples = usable[usable.split=='train'].groupby('method').head(1)
fig, axes = plt.subplots(1, len(examples), figsize=(18,3))
for ax, row in zip(np.atleast_1d(axes), examples.itertuples()):
    ax.imshow(Image.open(row.crops[0])); ax.set_title(f'{row.method}\nlabel={row.label}'); ax.axis('off')
plt.tight_layout(); plt.show()
# %% [markdown]
# Video batches: average frame logits before the video-level loss. Training uses
# random crops and modest augmentation; validation uses all cached crops.
# Balanced sampling is training-only. Each fake method receives equal weight.
# %%
class VideoFaces(Dataset):
    def __init__(self, frame, training=False):
        self.rows = frame.to_dict('records'); self.training=training
        self.augment = transforms.Compose([transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.1)])
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        paths = row['crops']
        if self.training: paths = random.sample(paths, min(TRAIN_FRAMES,len(paths)))
        images = []
        for path in paths:
            with Image.open(path) as image: image = image.convert('RGB')
            if self.training: image = self.augment(image)
            images.append(runtime.face_tensor(np.asarray(image)))
        return torch.stack(images), float(row['label']), row['uid']

def collate_videos(batch):
    faces, labels, ids = zip(*batch)
    return torch.cat(faces), [len(x) for x in faces], torch.tensor(labels, dtype=torch.float32), ids

def seed_worker(worker_id):
    worker_seed = torch.initial_seed()%2**32
    np.random.seed(worker_seed); random.seed(worker_seed)

train_frame = usable[usable.split=='train']
counts = train_frame.method.value_counts()
weights = [(.5/counts['original'] if r.method=='original' else .5/len(METHODS)/counts[r.method]) for r in train_frame.itertuples()]
generator = torch.Generator().manual_seed(SEED)
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)
loader_args = dict(batch_size=BATCH_VIDEOS, num_workers=WORKERS, pin_memory=True,
                   collate_fn=collate_videos, worker_init_fn=seed_worker)
train_loader = DataLoader(VideoFaces(train_frame, True), sampler=sampler, **loader_args)
val_loader = DataLoader(VideoFaces(usable[usable.split=='val']), shuffle=False, **loader_args)

def video_logits(model, faces, sizes):
    values = model(faces).flatten()
    return torch.stack([part.mean() for part in torch.split(values, sizes)])

@torch.inference_mode()
def evaluate(model, loader):
    model.eval(); labels, scores, ids = [], [], []
    for faces, sizes, y, uid in loader:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits = video_logits(model, faces.to(DEVICE, non_blocking=True), sizes)
        labels.extend(y.tolist()); scores.extend(logits.float().sigmoid().cpu().tolist()); ids.extend(uid)
    return pd.DataFrame(dict(uid=ids, label=np.asarray(labels,dtype=int), fake_score=scores))
# %% [markdown]
# Train and resume. The full best model is saved; Kaggle's preinstalled GPU Torch
# is retained. Frozen layers' batchnorm statistics stay frozen during both stages.
# Save notebook outputs before a session expires. To resume in a NEW session,
# attach the previous output, copy its deepguard_ff directory to /kaggle/working,
# then rerun cells in order with the same dataset/configuration. Input paths must match.
# %%
LAST = WORK/'last_checkpoint.pt'; BEST = WORK/'best_model.pt'
run_spec = dict(preprocessing=CFG, train_frames=TRAIN_FRAMES, head_epochs=HEAD_EPOCHS,
                finetune_epochs=FINETUNE_EPOCHS, batch_videos=BATCH_VIDEOS,
                seed=SEED, architecture='efficientnet_b0', learning_rates=[1e-3,1e-5,1e-4])
# Include exact split membership and cached crop counts, but not absolute paths.
fingerprint_text = usable[['uid','label','split','faces']].sort_values('uid').to_csv(index=False)
fingerprint = hashlib.sha256((fingerprint_text+json.dumps(run_spec,sort_keys=True)).encode()).hexdigest()
resume = torch.load(LAST, map_location='cpu', weights_only=True) if LAST.exists() else None
if resume:
    assert resume['fingerprint']==fingerprint, 'Dataset/settings changed: use a new WORK folder, not this checkpoint.'
    assert BEST.exists(), 'Restore best_model.pt alongside last_checkpoint.pt.'
model = runtime.build_model(pretrained=resume is None).to(DEVICE)
if resume: model.load_state_dict(resume['model'])
scaler = torch.amp.GradScaler('cuda')
if resume: scaler.load_state_dict(resume['scaler'])
best_auc = float(resume['best_auc']) if resume else -1.
history = json.loads((WORK/'history.json').read_text()) if resume and (WORK/'history.json').exists() else []
stages = [('head', HEAD_EPOCHS), ('finetune', FINETUNE_EPOCHS)]
for stage_index, (stage, epochs) in enumerate(stages):
    if resume and stage_index < resume['stage']: continue
    if stage=='finetune' and not (resume and resume['stage']==stage_index):
        model.load_state_dict(torch.load(BEST,map_location='cpu',weights_only=True))
    for param in model.parameters(): param.requires_grad=False
    for param in model.classifier.parameters(): param.requires_grad=True
    if stage=='finetune':
        for block in list(model.features.children())[-3:]:
            for param in block.parameters(): param.requires_grad=True
        backbone_params = [p for p in model.features.parameters() if p.requires_grad]
        groups = [dict(params=backbone_params, lr=1e-5), dict(params=model.classifier.parameters(), lr=1e-4)]
    else:
        groups = [dict(params=model.classifier.parameters(), lr=1e-3)]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    start, stale = 0, 0
    if resume and stage_index==resume['stage']:
        optimizer.load_state_dict(resume['optimizer'])
        start, stale = resume['epoch']+1, resume['stale']
        generator.set_state(resume['sampler_rng'])
    for epoch in range(start, epochs):
        if stage=='finetune' and stale>=PATIENCE: break
        started=time.time(); model.eval()
        model.classifier.train()
        if stage=='finetune':
            for block in list(model.features.children())[-3:]: block.train()
            for module in model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm): module.eval()
        total_loss, seen = 0., 0
        for batch_index, (faces, sizes, labels, ids) in enumerate(train_loader):
            faces=faces.to(DEVICE, non_blocking=True); labels=labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits=video_logits(model, faces, sizes)
                loss=nn.functional.binary_cross_entropy_with_logits(logits, labels)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss. Do not export; inspect inputs/learning rate.')
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            scaler.step(optimizer); scaler.update()
            total_loss+=float(loss.detach())*len(labels); seen+=len(labels)
            if (batch_index+1)%100==0:
                print(f'{stage} epoch {epoch+1}: batch {batch_index+1}/{len(train_loader)}, loss {total_loss/seen:.4f}', flush=True)
        prediction=evaluate(model,val_loader)
        auc=float(roc_auc_score(prediction.label,prediction.fake_score))
        # Per-stage stopping; global best checkpoint always uses validation AUC.
        stage_previous=[r['val_auc'] for r in history if r['stage']==stage]
        stale=0 if not stage_previous or auc>max(stage_previous) else stale+1
        if auc>best_auc:
            best_auc=auc
            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()}, BEST)
        record=dict(stage=stage, epoch=epoch+1, train_loss=total_loss/seen,
                    val_auc=auc, seconds=time.time()-started)
        history.append(record); print(record, flush=True)
        (WORK/'history.json').write_text(json.dumps(history,indent=2))
        checkpoint=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                        optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                        stage=stage_index, epoch=epoch, stale=stale, best_auc=best_auc,
                        fingerprint=fingerprint, sampler_rng=generator.get_state())
        temporary=WORK/'last_checkpoint.tmp'
        torch.save(checkpoint,temporary); temporary.replace(LAST)
    resume=None
model.load_state_dict(torch.load(BEST,map_location='cpu',weights_only=True)); model.eval()
print('Best validation AUC:', best_auc)
print('Saved resumable checkpoint:', LAST)
# %% [markdown]
# Select a binary threshold using VALIDATION ONLY. Use a fixed +/-0.10 uncertain
# band around it. This band is a heuristic, not calibrated confidence or OOD detection.
# Do not rerun model selection in response to final test results.
# %%
val_prediction=evaluate(model,val_loader)
thresholds=np.linspace(.1,.9,161)
quality=[balanced_accuracy_score(val_prediction.label,val_prediction.fake_score>=t) for t in thresholds]
best_indices=np.flatnonzero(np.isclose(quality,max(quality)))
chosen=min(best_indices,key=lambda i:abs(thresholds[i]-.5))
threshold=float(thresholds[chosen]); low=max(0.,threshold-.1); high=min(1.,threshold+.1)
val_prediction.to_csv(WORK/'validation_predictions.csv',index=False)
print('Validation binary threshold:',threshold,'uncertain band:',(low,high))
# %% [markdown]
# FINAL TEST: report all methods, both classes, clear-answer coverage and input
# rejection rates. These are in-dataset results, not unknown-source guarantees.
# Group bootstrap resamples official source pairs to reflect related fake videos.
# %%
test_loader=DataLoader(VideoFaces(usable[usable.split=='test']),shuffle=False,**loader_args)
test_prediction=evaluate(model,test_loader)
test_prediction=test_prediction.merge(usable[['uid','method','source_ids']],on='uid',validate='one_to_one')
y=test_prediction.label.to_numpy(); scores=test_prediction.fake_score.to_numpy(); binary=scores>=threshold
clear=(scores<=low)|(scores>=high)
test_prediction['decision']=np.where(scores<=low,'Likely real',np.where(scores>=high,'Likely fake','Not sure'))
test_prediction.to_csv(WORK/'test_predictions.csv',index=False)
# Build connected source groups from retained test videos.
parent={v:v for ids in test_prediction.source_ids for v in ids}
def find(v):
    while parent[v]!=v:
        parent[v]=parent[parent[v]]; v=parent[v]
    return v
for ids in test_prediction.source_ids:
    for v in ids[1:]: parent[find(v)]=find(ids[0])
groups=np.array([find(ids[0]) for ids in test_prediction.source_ids]); unique=np.unique(groups)
rng=np.random.default_rng(SEED); boot=[]
for _ in range(500):
    draw=rng.choice(unique,len(unique),replace=True)
    indices=np.concatenate([np.flatnonzero(groups==g) for g in draw])
    if len(np.unique(y[indices]))==2:
        boot.append(balanced_accuracy_score(y[indices],binary[indices]))
cm=confusion_matrix(y,binary,labels=[0,1])
metrics=dict(roc_auc=float(roc_auc_score(y,scores)), accuracy=float(accuracy_score(y,binary)),
             balanced_accuracy=float(balanced_accuracy_score(y,binary)),
             confusion_matrix=cm.tolist(), confusion_matrix_order=['real','fake'],
             fake_recall=float(cm[1,1]/cm[1].sum()), real_false_positive_rate=float(cm[0,1]/cm[0].sum()),
             binary_threshold=threshold, real_threshold=low, fake_threshold=high,
             test_total=int((prepared.split=='test').sum()), test_analyzed=len(y),
             input_coverage=float(len(y)/(prepared.split=='test').sum()),
             clear_answers=int(clear.sum()), clear_answer_fraction=float(clear.mean()),
             clear_answer_accuracy=float(accuracy_score(y[clear],binary[clear])) if clear.any() else None,
             balanced_accuracy_group_bootstrap_95ci=np.percentile(boot,[2.5,97.5]).tolist() if boot else None,
             independent_source_groups=len(unique),
             scope='FF++ c23 official-source split; no external-source accuracy established')
print(json.dumps(metrics,indent=2))
print(classification_report(y,binary,target_names=['Real','Fake'],zero_division=0))
method_rows=[]
for method, frame in test_prediction.groupby('method'):
    predicted=frame.fake_score.to_numpy()>=threshold
    method_rows.append(dict(method=method,videos=len(frame),correct_fraction=float((predicted==frame.label.to_numpy()).mean())))
print(pd.DataFrame(method_rows).to_string(index=False))
(WORK/'metrics.json').write_text(json.dumps(metrics,indent=2))
pd.DataFrame(method_rows).to_csv(WORK/'method_results.csv',index=False)
# %% [markdown]
# Export the COMPLETE fine-tuned model and the matching standalone local predictor.
# The existing DeepGuard JSON-head loader cannot load this checkpoint directly.
# Use the included local app/CLI. Integration into the existing app is a separate change.
# %%
EXPORT=WORK/'export'; EXPORT.mkdir(exist_ok=True)
shutil.copy2(BEST,EXPORT/'model.pt')
for name in list(runtime.ASSETS)+['OPENCV_LICENSE','RES10_UPSTREAM_README.md']:
    shutil.copy2(ASSET_DIR/name,EXPORT/name)
shutil.copy2(WORK/'deepguard_ff_runtime.py',EXPORT/'deepguard_ff_runtime.py')
config=dict(schema=1,architecture='efficientnet_b0',label_mapping={'real':0,'fake':1},
            preprocessing=CFG,aggregation='mean_frame_logits_then_sigmoid',
            crop_cache_jpeg_quality=95,real_threshold=low,fake_threshold=high,binary_threshold=threshold,
            model_sha256=hashlib.sha256((EXPORT/'model.pt').read_bytes()).hexdigest(),
            run_fingerprint=fingerprint,training=run_spec,
            versions=dict(torch=str(torch.__version__),torchvision=str(torchvision.__version__),opencv=cv2.__version__),
            score_notice='Classifier score, not probability of authenticity; no external-source validation yet.')
(EXPORT/'config.json').write_text(json.dumps(config,indent=2))
for filename in ['metrics.json','method_results.csv','history.json','coverage.csv','validation_predictions.csv','test_predictions.csv','train.json','val.json','test.json']:
    shutil.copy2(WORK/filename,EXPORT/filename)
public_manifest=manifest.drop(columns=['path']).copy()
public_manifest.to_csv(EXPORT/'split_manifest.csv',index=False)
# Local Streamlit app uses the same preprocessing and weights; all video handles close.
(EXPORT/'app.py').write_text('''from pathlib import Path
import tempfile
import streamlit as st
from deepguard_ff_runtime import predict_video
st.set_page_config(page_title="DeepGuard FF++", page_icon="🎥")
st.title("Check a video")
st.write("Looks for learned signs of facial manipulation. Audio is not used.")
st.caption("Research model trained on FaceForensics++ c23. Results are not proof of authenticity.")
video=st.file_uploader("Choose a video up to 100 MB",type=["mp4","avi","mov","mkv"])
if video is not None:
    st.video(video.getvalue())
    if st.button("Check this video",type="primary"):
        if video.size>100*1024**2:
            st.error("Choose a video under 100 MB.")
        else:
            try:
                with st.spinner("Reading the video and checking face samples..."):
                    with tempfile.TemporaryDirectory(prefix="deepguard-ff-") as folder:
                        path=Path(folder)/("upload"+Path(video.name).suffix.lower())
                        path.write_bytes(video.getvalue())
                        result=predict_video(path,Path(__file__).resolve().parent)
                st.subheader(result["decision"])
                st.metric("Face samples checked",result["faces_found"])
                if result["fake_score"] is not None:
                    st.metric("Fake-pattern score",f"{result['fake_score']:.1%}")
                if result["decision"]=="Not sure":
                    st.info("Analysis completed, but the score is in the uncertain range. Rechecking the same video will not improve the model.")
                if result.get("reason"):st.warning(result["reason"])
                st.caption(result["notice"])
                st.json(result,expanded=False)
            except Exception as exc:
                st.error("The check could not finish. Inspect the details below.")
                st.exception(exc)
''')
(EXPORT/'requirements-local.txt').write_text('torch==2.6.0\ntorchvision==0.21.0\nnumpy==2.1.3\nopencv-contrib-python==4.11.0.86\nPillow==11.1.0\nstreamlit==1.43.2\n')
(EXPORT/'README.txt').write_text('Use Python 3.12 and a fresh virtual environment. See the accompanying Kaggle guide.\nCLI: python deepguard_ff_runtime.py your_video.mp4\nUI: python -m streamlit run app.py --server.fileWatcherType none --server.maxUploadSize 100\nCPU is supported. No dataset or ImageNet download is needed for inference.\nThis checkpoint does not fit the old visual.json loader.\n')
# Confirm exported CPU runtime matches notebook predictions on one VALIDATION video.
example=usable[usable.split=='val'].iloc[0]
local_result=runtime.predict_video(example.path,EXPORT,device='cpu')
expected=float(val_prediction.loc[val_prediction.uid==example.uid,'fake_score'].iloc[0])
assert local_result['fake_score'] is not None and abs(local_result['fake_score']-expected)<.005, (local_result,expected)
print('Notebook/CPU export parity passed:',local_result['fake_score'],expected)
archive=shutil.make_archive('/kaggle/working/deepguard_ff_export','zip',EXPORT)
print('Download from notebook Output:',archive)
print('ALSO save last_checkpoint.pt and the working folder for resume; they are not in the inference ZIP.')
# %% [markdown]
# Finish: use Save Version with outputs to preserve the export ZIP. Download ONLY
# deepguard_ff_export.zip for local inference, not the attached 18 GB dataset.
# Preserve deepguard_ff/ separately if you need to resume training.
# Run cells 1..end once; keep the final test results unchanged when reporting.
