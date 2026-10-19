# Reproduce expanded video-only training

From the checkout with the owner's authorized local files:

    bash scripts/train.sh data/manifests/balanced_pending.csv --balanced
    bash scripts/validate.sh

The helper verifies hashes, labels and readability, preserves previous source-family partitions and assigns only new families. Training checks hashes again before reading cached features. Raw files remain outside Git.

The retry preprocessing uses the bundled OpenCV ResNet-10 SSD face detector, confidence 0.9 and NMS 0.3. Twenty-four whole-clip positions are sampled sequentially. Single or clearly dominant foreground face crops with 15% context are resized to 224 pixels. The minimum is four face samples and 15% facial coverage; a foreground box must have at least four times the area of the next box to be accepted among background faces. Comparable faces and blur still cause rejection. The detector, sampler, sampling count and dominance policy are included in the cache key, so older features cannot be silently reused.

Verified official ImageNet EfficientNet-B0 weights generate frozen 1,280-dimensional frame embeddings. Mean and standard deviation plus eight texture values form a 2,568-dimensional video vector. Only the classifier is fitted; no full backbone fine-tuning or GPU training is claimed. This is CPU transfer learning.

After quality filtering, each split has equal Real/Fake counts. Training originals and augmentations are balanced separately. Validation, calibration and testing contain originals only. Training weights give each source family equal total weight within its class rather than multiplying evidence for families with more augmentations.

Seven classifiers are compared using validation ROC-AUC, with validation Brier score as tie-breaker:

- Logistic heads with C = .001, .01, .1, 1 and 10.
- Neural heads with 32 ReLU units, dropout .2 and AdamW weight decay .01 or .1.

The scaler is fitted on training rows only. Neural heads clip standardized features to [-10,10], use learning rate .001, train for at most 150 epochs, check validation every five epochs and stop after eight checks without improvement. The best validation state is exported. Seed 42 is fixed. Test labels never select classifier type, regularization or epoch. V004 preprocessing fixes were motivated by inspected reserved-video failures; its evaluation is regression monitoring rather than a newly blinded test.

Temperature scaling is fitted on separate calibration originals and evaluated separately. The UI displays the raw score, not a probability of truth. The JSON artifact supports linear or neural parameters; no pickle is loaded.

Primary evaluation uses reserved originals from new source families. Historical test originals have a separate monitoring report because previous results were already inspected. Compression/blur/darkness variants are robustness tests on those sources, not an independent external dataset.

reports/results/model_selection.json records validation comparisons and neural training histories. summary.json contains actual counts and primary metrics. split_membership.csv records quality/balance decisions. checkpoints/visual.json contains the fitted model. The app invalidates cached resources when the model changes.

## Two-pass deployment (V004)

The standard model retains its original 16-frame preprocessing and fitted parameters.
The retry head was retrained separately on 24-frame sequential inputs with the
dominant-face policy in configs/retry.yaml. Reproduce the components separately:

```bash
deepguard train data/manifests/splits.csv --balanced --model checkpoints/visual.json --results reports/results
deepguard --config configs/retry.yaml train data/manifests/splits.csv --balanced --model checkpoints/coverage_retry.json --results reports/retry_training
deepguard robustness
```

The default config routes only insufficient-evidence failures to the retry head.
Existing scored standard results are preserved. The retry model was not promoted
as a global replacement because that regressed on inspected reserved videos.
Read EVALUATION_REPORT.md for the actual pipeline metrics and the distinction
between standard and retry fitting counts. Reused reserved files provide monitoring,
not fresh external validation. The private prepared.csv is overwritten by each
component training pass; component selection reports retain their own memberships.
