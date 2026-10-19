# Video-only application architecture

    Video upload -> temporary file -> FFprobe validation
      -> whole-clip frame sampling -> neural SSD face detector
      -> quality checks -> frozen EfficientNet-B0 face embeddings
      -> mean/std + texture -> training-only fitted scaler
      -> validation-selected linear or neural classifier
      -> raw score + uncertainty gate -> plain-language result and report

The temporary video is deleted after analysis. Session previews and extracted images remain until the user clears the session or it ends. No audio features are used.

media.py probes size, duration and dimensions. The standard model retains its original 16-frame sampling and quality gates. Only insufficient evidence triggers a separate 24-frame sequential search with a dominant foreground face policy (area ratio at least four), a four-crop minimum and 15% facial coverage. Comparable faces and blur remain quality failures. Each pass uses a classifier trained with that exact preprocessing. SSD detector assets are checksummed; legacy Haar support exists only for explicitly configured artifacts.

models.py verifies official EfficientNet weights, aggregates 2,568 visual values, validates safe JSON classifier arrays and computes linear or neural predictions. selection.py fits training-only scaling and compares seven candidate classifiers on validation data, with source-family weights.

data.py validates labels and hashes, groups potential original/fake and augmentation families, preserves earlier split assignments, and balances quality-eligible files. Augmentations from reserved source families never enter fitting. Original-only validation, calibration and test rows are selected without model scores.

training.py fits and exports the actual model, calibration, per-video predictions, baseline comparisons, quality exclusions and balance decisions. The new-source test cohort is the primary report; older previously inspected test videos are monitored separately. Model type, regularization and neural epoch are selected using validation only.

inference.py applies the same preprocessing and classifier, with feature-distance and score uncertainty. reports/results provides reproducible metrics and public split membership without raw paths. app/main.py exposes six plain-language pages, actual measurements, face evidence and a downloadable report. The UI checks model/preprocessing version compatibility before allowing a check.

The trained classifier, detector assets and generic pretrained encoder weights suffice for inference. Original training videos and private manifests are not inference dependencies. configs/default.yaml and the model must describe the same preprocessing.
