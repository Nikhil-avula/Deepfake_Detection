# Demonstrate the updated application

1. Start the app with bash scripts/run.sh in the cloud checkout, or python -m streamlit run app/main.py in your activated local environment.
2. Open Start and explain the video-only objective. Use Check a video to upload a short, clearly visible single-person clip.
3. Explain the actual result: Likely real, Likely fake, Not sure or Cannot analyze. Show the face samples and score explanation. Download the HTML report if useful.
4. Use a held-out original for a meaningful dataset check. A training-file warning means the clip is not an independent test. Renaming a file cannot change its score.
5. Open How reliable is it? and quote the current measured accuracy, number of original test videos, source groups and coverage. Do not substitute an older experiment's metrics or claim unknown-video accuracy.
6. Open Our dataset: candidate Real/Fake counts are equal, and post-quality per-split counts are displayed. Explain that augmentations stay with their original family and are not independent examples.
7. Show limitations and robustness results. A compressed version of the same reserved clip is a robustness check, not an external dataset.
8. Use Clear video and result to remove retained evidence. Test malformed or faceless input to show an understandable failure result rather than a fabricated classification.

The selected model is recorded in reports/results/model_selection.json. EfficientNet supplies frozen features; validation chooses a regularized linear or small neural head. Only the classifier is fitted on the owner's examples.
