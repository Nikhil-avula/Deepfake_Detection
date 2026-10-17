# Ten-minute presentation and honest demonstration

**0:00–1:00 — Motivation.** Explain manipulated faces versus synthetic voices and why
one detector can miss the other. Use the abstract's fraud example without claiming the
tool prevents fraud reliably.

**1:00–2:00 — Research question.** State the quality/availability-aware fusion hypothesis.
Explain that a new interface alone is not the research contribution.

**2:00–3:00 — Architecture.** Walk through ARCHITECTURE.md: permission, labels, connected
splits, four modalities, fusion and separate calibration/test partitions.

**3:00–4:00 — Data and ethics.** Show Dataset Explorer. Currently there are no local
manifests. Explain FakeAVCeleb approval and why overall labels cannot supervise every
modality. State the access blocker plainly.

**4:00–5:00 — Models.** Show Model Information: currently “Model not trained”. Explain
EfficientNet, spectrogram CNN, frozen-embedding LSTM and the learned alignment task.

**5:00–6:30 — Software behavior.** Launch Streamlit with the README command. Upload an
authorized local video if available, or describe the generated decoding fixture as a
software check. No checkpoint means no real/fake prediction; show this refusal honestly.
Do not stage a synthetic fixture as a real research example.

**6:30–7:30 — Evidence and report.** If a real upload was analyzed, show actual extracted
frames/spectrogram, missing-face/audio warnings and downloadable HTML. No Grad-CAM or
sync score is shown without the relevant trained model/asset.

**7:30–8:30 — Verification.** Show PROJECT_STATUS and the actual saved pytest summary.
Explain that engineering tests differ from held-out detection metrics. Research
Laboratory currently contains no benchmark results.

**8:30–9:30 — Experimental plan.** Explain reserved fusion/calibration partitions,
unseen-method testing, external visual/audio tests, ablations and paired degradation.
Use planned commands, never present them as completed runs.

**9:30–10:00 — Limitations and next step.** Permission and landmark asset access first,
then audited metadata, genuine training and evaluation. Conclude with the research
question, not a fabricated accuracy claim. After actual experiments, update this script
with the exact artifact paths and verified results rather than illustrative numbers.
