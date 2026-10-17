# Viva preparation: 45 questions and answers

1. **What problem are you solving?** We investigate whether face, voice, motion and
   synchronization evidence together improve deepfake detection on held-out media.
2. **What is a deepfake?** Synthetic or manipulated media that changes apparent identity,
   appearance or speech; the exact label depends on the dataset's documented task.
3. **Why use several modalities?** One can be genuine while another is manipulated.
   Complementarity must be measured rather than assumed.
4. **What is the current research result?** None yet: data permission and training are
   pending. Engineering tests verify software behavior, not detection accuracy.
5. **What does your abstract require?** Visual, audio, temporal and lip-sync analysis,
   baselines, fusion, an upload interface and unseen-method/dataset evaluation.
6. **What is the proposed contribution?** A controlled study of quality- and
   availability-aware late fusion against fair unimodal and fusion comparisons.
7. **Is a dashboard scientific novelty?** No. It helps inspect evidence but does not
   establish a new detection method or empirical improvement.
8. **Why FakeAVCeleb?** Its authors document audio/video manipulation categories useful
   for multimodal targets; actual release suitability still needs an approved audit.
9. **Why not download it immediately?** Owners require approval. Public documentation is
   not dataset permission.
10. **Why external Celeb-DF and ASVspoof?** They can test visual and audio transfer,
    respectively, with compatible labels; they cannot be pooled as one benchmark.
11. **Can you assign audio labels from video labels?** No. A real voice may accompany a
    manipulated face. Unknown modality labels are explicitly -1.
12. **What is leakage?** Training information contaminating evaluation, such as the same
    identity, source recording or derivative appearing in multiple splits.
13. **How do you prevent derivative leakage?** Union-find groups all shared donor/target
    identities and source recordings, transitively, plus duplicate file hashes.
14. **Why five partitions?** Train optimizes base models, val selects them, fusion fits
    combination weights, calibration fits temperature, and test measures final behavior.
15. **Does grouping ensure class balance?** No. Leakage safety takes priority; inspect
    counts and predeclare any redesign instead of tuning test splits.
16. **What is a CNN?** A neural network with shared local filters that learn spatial
    patterns, useful for face images and time–frequency audio maps.
17. **Why EfficientNet?** It provides a compact established visual backbone; suitability
    still requires comparison and profiling.
18. **Does ImageNet pretraining detect deepfakes?** No. It initializes generic visual
    representations; the deepfake task still needs labeled training.
19. **How are frames aggregated?** Average their logits at video level, then apply the
    sigmoid. Average logits and average probabilities are different choices.
20. **What is a Mel-spectrogram?** A time–frequency energy representation using
    perceptually spaced frequency bands, usually log-scaled for dynamic range.
21. **Why normalize with a fixed range?** It avoids fitting normalization statistics on
    validation/test data. The range is a declared engineering choice needing validation.
22. **Does the energy threshold detect speech?** No. It removes obvious silence but can
    admit music/noise. A validated VAD is still pending.
23. **What is an LSTM?** A recurrent network with gates and memory that processes ordered
    embeddings and can learn temporal dependencies.
24. **Why freeze the visual encoder for temporal training?** It controls computation and
    helps isolate the value of temporal modeling using the same representation.
25. **How do mouth features arise?** A genuine landmark model locates mouth points and
    crops the corresponding region. No landmark asset means no claimed mouth evidence.
26. **What does the synchronizer learn?** Whether mouth/audio windows are aligned or
    shifted within a genuine recording, using two trainable convolution streams.
27. **Why within-recording negative pairs?** They reduce simple identity/recording
    shortcuts that arise when unrelated people supply the negative pairs.
28. **Does sync mismatch prove fakery?** No. Dubbing, editing and recording delays are
    alternative explanations, so mismatch is evidence with explicit limitations.
29. **How do you handle timestamp offsets?** Subtract audio-start minus video-start from
    video-relative window centers before indexing the decoded waveform.
30. **Why Logistic Regression, Random Forest and XGBoost?** They provide interpretable
    linear and nonlinear baselines on declared texture/audio/motion features.
31. **Why fit scaling only on train?** Statistics from evaluation data leak information
    into model development and can inflate performance.
32. **What is late fusion?** Combining outputs after individual modality processing.
    Here learned fusion also receives quality and availability indicators.
33. **How are absent modalities represented?** Internally with a neutral placeholder
    and availability zero; externally they remain absent, with no invented score.
34. **Why exclude raw sync from direct averaging?** Alignment mismatch and fake-class
    probability have different meanings. Learned fusion can model their association.
35. **What does reliability weighting do?** It averages model log-odds using quality
    weights. Those heuristic weights may be biased and require comparison.
36. **What is calibration?** Agreement between predicted scores and observed frequency
    on a defined distribution. Temperature scaling fits a positive logit scale.
37. **Does calibrated confidence guarantee truth?** No. Calibration can fail under
    domain shift and is not a guarantee about an individual video's authenticity.
38. **Why return Inconclusive?** Weak/conflicting evidence or a score near the decision
    boundary does not justify a confident classification.
39. **Why is accuracy insufficient?** Class imbalance can hide poor fake detection or
    excessive false alarms; precision, recall, AUCs and confusion rates add context.
40. **How do ROC-AUC and PR-AUC differ?** ROC compares true/false positive rates across
    thresholds; PR compares precision/recall and emphasizes positive-class behavior.
41. **What are Brier and ECE?** Brier is mean squared probability error; ECE is a binned
    score-versus-frequency discrepancy whose value depends on binning/sample size.
42. **What is an ablation?** Removing a component to study its contribution. Inference
    removal differs from retraining without the component; report which was done.
43. **What does Grad-CAM prove?** Nothing about authenticity by itself; it highlights
    spatial features that influence a particular model logit.
44. **How is robustness tested?** Apply predeclared paired degradations to untouched
    held-out sources and evaluate unchanged weights, retaining group relationships.
45. **Who performed the work and what remains?** The initial implementation is
    AI-assisted. Team members must record actual reviews, experiments and explanations;
    permissions, real training, calibration and unseen-data evaluation remain pending.
