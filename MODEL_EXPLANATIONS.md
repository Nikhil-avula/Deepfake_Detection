# Visual model explanation

The application separates face localization from manipulation classification. OpenCV's ResNet-10 SSD locates faces; it does not decide whether they are fake. Its bundled public weights and network definition are checksummed. The standard pass retains its original sixteen samples and quality checks. If it cannot obtain sufficient face evidence, a separately trained retry classifier uses twenty-four sequential samples, a four-crop minimum, 15% coverage and a foreground face area ratio of at least four. A scored standard result, including Not sure, is never replaced.

A frozen official ImageNet EfficientNet-B0 encodes each cropped face into 1,280 values. Mean and standard deviation across the video, plus eight texture statistics, form 2,568 features. ImageNet pretraining supplies general visual representations, not prior certification that a face is authentic.

A scaler fitted only on training videos standardizes those features. Five regularized logistic classifiers and two small neural classifiers are trained and compared on validation originals. The neural alternative has 32 ReLU units, training-only dropout and AdamW optimization. The best validation checkpoint is exported, with all parameters stored as ordinary JSON arrays. The chosen architecture is recorded in the model artifact and model-selection report; a more complex candidate is not automatically preferred.

Source-family weights prevent numerous augmentations of one original from dominating training. Real and Fake file counts are equal after quality filtering. Augmentations are used only in the learning partition, not as independent validation, calibration or primary test examples.

A separate calibration partition fits temperature scaling. Raw and scaled results are measured separately. The app displays raw fake-pattern scores, not probabilities of truth. Scores between 0.35 and 0.65 give Not sure. A heuristic feature-distance gate also returns Not sure on unusually different visual features; this is not a validated guarantee of out-of-distribution detection.

No filename or dataset label is used to generate a prediction. Exact-file hashes only add a warning when an analyzed file was in training or held-out testing. Renaming a video must leave its score unchanged. Audio is not analyzed.

Fine-tuning every image-network layer would require more independent examples and additional compute. The implemented approach is supervised transfer learning of a compact classifier, with the image encoder frozen. Evaluation on genuinely external videos is still needed.
