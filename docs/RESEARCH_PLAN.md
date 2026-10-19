# Video-only research plan

Problem: distinguish real-labeled and face-manipulated videos using visual face samples,
then communicate evidence and uncertainty clearly to a nontechnical user.

Question: how much separation do generic pretrained visual representations provide on a
small owner-labeled source-pair test? Compare a frozen EfficientNet feature head with
texture-only logistic and forest baselines on identical sources. Novelty is not claimed
for an established architecture or interface; the contribution is a transparent,
reproducible student experiment with documented failures and picture-quality robustness.

Hypotheses: transfer features improve held-out ranking over simple texture; compression
can degrade detection; abstention can trade coverage for reduced risk. Results may
contradict these hypotheses and must remain reported. A high AUC alone is not a promise
of accurate operational decisions.

Success criteria: no source-pair leakage, real trained artifacts, real per-video test
predictions, explicit exclusions/coverage, reproducible commands and understandable UI.
The current measured test accuracy alone cannot establish a reliable external detector.
More data with known origin, permissions, subjects and manipulation methods are needed.

Next independent study: acquire genuinely new licensed subjects/methods; lock them away
before further tuning; improve face detection and verify identity/source groups; compare
predeclared methods, score calibration and coverage. Do not manufacture fake labels,
copy test videos into training, hide difficult cases or report synthetic checks as accuracy.
