# The visual model, explained simply

The model learns from pictures of faces extracted from labeled videos. It does not
read filenames as prediction features and does not classify by matching a file to a
stored real/fake label.

A CNN learns image patterns using shared filters. EfficientNet-B0 is an established
image backbone (Tan and Le, 2019, https://arxiv.org/abs/1905.11946). We use its official
ImageNet initialization and freeze its weights, then fit our own classifier. This is
transfer learning: we trained a decision head, not the whole neural network.

Each frame becomes 1,280 numerical features. Their video-level mean and variation plus
eight measured texture statistics form 2,568 features. Training-only StandardScaler
normalizes them; a regularized Logistic Regression head learns the real/fake distinction.
Validation chooses regularization. A texture-only Logistic Regression and Random Forest
serve as simpler comparison baselines on the same held-out sources.

A sigmoid transforms the classifier logit into a model score. Temperature scaling
(Guo et al., 2017, https://arxiv.org/abs/1706.04599) is fitted on a separate small
calibration split and evaluated separately. It did not establish trustworthy confidence
on this pilot. The application shows an uncalibrated classifier score, never the
probability that a video is truly fake.

Likely real/fake is issued outside a predeclared 0.35–0.65 score band only when facial
quality is adequate and a heuristic feature-distance gate does not flag unfamiliarity.
The distance gate is not a validated universal out-of-distribution detector. Not sure
and Cannot analyze are legitimate results, not forced fake or real predictions.

OpenCV handles frames and baseline face detection; PyTorch/torchvision provide the
feature backbone; Scikit-learn fits classifiers and computes metrics; Streamlit provides
a readable local interface. The model can mistake person, generation or compression
patterns for forgery cues. A face crop is evidence input, not proof of a forged region.
No validated pixel-level localization or identity recognition is claimed.
