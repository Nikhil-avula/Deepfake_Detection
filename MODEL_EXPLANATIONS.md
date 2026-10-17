# Model explanations

**CNN:** learns local patterns through shared filters. Faces and spectrograms have
spatial structure; convolution can represent texture and frequency/time patterns.

**EfficientNet-B0:** a compact image architecture balancing depth, width and resolution
([Tan and Le, 2019](https://arxiv.org/abs/1905.11946)). A linear head learns visual
manipulation from face crops. ImageNet initialization is optional and is not itself a
trained deepfake detector. Video evidence averages frame logits; repeated frames are
not independent test samples.

**Log-Mel spectrogram:** short overlapping Fourier windows summarize audio energy in
perceptually spaced frequency bands. Fixed log scaling avoids fitting normalization on
test data. A small CNN predicts audio spoof labels, not video labels. Silence, music,
language and codecs can all create domain shift. RMS gating does not establish speech.

**LSTM:** maintains a recurrent state over ordered visual embeddings. The EfficientNet
encoder is frozen for temporal training to isolate sequence modeling. Eight or more
timestamp-consistent sampled frames are required; missing frames can invalidate a clip.
Whether temporal modeling adds value is an experiment, not an assumption.

**Synchronization:** actual landmark mouth crops and audio windows share video-relative
timestamps after audio-start correction. A two-stream 3D CNN learns aligned/shifted pairs
from genuine recordings. Negative pairs are within the same clip to limit identity
shortcuts. It is inspired by the task studied in
[Chung and Zisserman, Out of Time](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/),
not an implementation or pretrained checkpoint claim for SyncNet. This exact network is
unvalidated until held-out offset experiments run. Dubbing, natural delay and encoding
offsets can cause mismatch without synthesis.

**Classical baselines:** Logistic Regression is linear and interpretable; Random Forest
models nonlinear interactions through trees; XGBoost fits successive error-correcting
trees. Inputs include brightness/contrast, edge/Laplacian texture, temporal pixel change,
Mel statistics and availability/quality. All imputation/scaling fits only train rows.
These are meaningful, limited baselines, not substitutes for trained multimodal models.

**Fusion:** simple averaging and quality-weighted log-odds exclude raw synchronization
mismatch. Learned logistic late fusion can learn the relationship between mismatch and
overall labels jointly with other evidence. Quality weighting is a hypothesis: quality
heuristics may themselves introduce bias. The twelve-feature fusion includes availability
flags so absent evidence differs from a neutral observed score.

**Confidence:** binary logits become scores through sigmoid. Temperature scaling divides
the final logit by a positive scalar fitted on a separate calibration set
([Guo et al., 2017](https://arxiv.org/abs/1706.04599)). Calibration is evaluated with Brier
score and ECE; fitting temperature alone does not guarantee calibration under domain
shift. The app exposes final confidence only for a separately calibrated fusion model,
and abstains within a policy band or under disagreement. Those thresholds still require
validation on actual data.

**Grad-CAM:** spatial attribution of the visual fake logit using final feature-map
gradients ([Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391)). It highlights regions
influencing this model output, not proof of manipulation or a reliable localization mask.

PyTorch provides inspectable models/autograd and CPU execution; OpenCV decodes frames
and detects baseline faces; FFmpeg extracts codecs/audio; Librosa produces spectral
features; Scikit-learn supports robust baseline and metric implementations; Streamlit
makes the research workflow inspectable without a separate frontend/backend stack.
Transformers, ResNet, a stronger detector/VAD and pretrained SyncNet were considered;
LSTM and a trainable two-stream alignment model keep dependencies and provenance simpler.
Quality and accuracy tradeoffs must still be measured.

Function/module responsibilities are explained in ARCHITECTURE.md and source docstrings.
Use `deepguard --help` and `deepguard COMMAND --help` to inspect each executable interface.
