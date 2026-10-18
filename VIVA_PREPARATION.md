# Video-only viva preparation: 45 questions and answers

1. **What is the current problem statement?** A user uploads a video and receives visual real/fake evidence with a clear uncertain or unsupported outcome.

2. **What does the system actually inspect?** Face images sampled across the full video, not the filename or a stored class lookup.

3. **What are the four analysis outcomes after setup?** Likely real, Likely fake, Not sure and Cannot analyze. The last two avoid forced guesses.

4. **What data were provided?** The owner supplied 49 real-labeled and 49 fake-labeled videos, 98 files total.

5. **Who supplied the labels?** The owner defined numbered files as real and _fake files as fake; label truth and benchmark origin were not independently verified.

6. **Why not randomly split frames?** Frames and derivatives of the same source could leak nearly identical information into evaluation.

7. **How are source pairs grouped?** Matching numbered filename stems remain together, and byte-identical content is also grouped.

8. **Are the splits identity-disjoint?** That is not established. Subject and donor metadata were absent, so only source-pair separation is claimed.

9. **What are the four data partitions?** Train fits the classifier, validation chooses settings, calibration fits a score transformation, and test measures reserved source groups.

10. **How many inputs are in each partition?** Before quality exclusions: 58 train, 12 validation, 12 calibration and 16 test.

11. **What happens after quality filtering?** The usable counts are 42 train, 9 validation, 12 calibration and 10 test; all exclusions are recorded.

12. **What is a CNN?** A network with shared local filters that can learn spatial image patterns.

13. **Why EfficientNet-B0?** It is an established compact image encoder with official pretrained initialization and an inspectable implementation.

14. **Did we train the entire backbone?** No. It stays frozen; we genuinely fit a new visual classifier on its video representations.

15. **What is transfer learning?** Reusing a generic learned representation and fitting a task-specific model using the current labels.

16. **What is the feature vector?** Mean and standard deviation of 1280 frame embedding features plus eight texture statistics, for 2568 values per video.

17. **Why include variation?** Variation summarizes changes between sampled face representations, though it is not a separately trained temporal sequence detector.

18. **What is Logistic Regression?** A regularized linear decision function converted by sigmoid into a class score.

19. **Why scale features using training data only?** Using evaluation statistics during fitting would leak information and make the procedure less honest.

20. **How was regularization chosen?** Five C values were compared on validation ROC-AUC, with validation Brier score as a tie-breaker.

21. **Why include simpler baselines?** Texture logistic and forest models show whether the more expensive representation adds value under the same split.

22. **Why sample across the whole clip?** Checking only the beginning can miss manipulated portions later in a video.

23. **How are faces detected?** OpenCV Haar detection supplies baseline single-face crops; it can miss poses or produce false boxes.

24. **When does quality filtering reject an input?** Too few clear faces, too many ambiguous multi-face samples or excessive blur can prevent useful evidence.

25. **Does rejection mean fake?** No. It means the system cannot support an authenticity conclusion for that input.

26. **What is a model score?** A numerical output reflecting the fitted classifier, not a guaranteed probability of truth.

27. **What is temperature scaling?** A positive logit scale fitted on a separate calibration partition; this small pilot did not establish trustworthy calibrated confidence.

28. **Which scores does the application display?** Raw classifier scores. Raw and temperature-scaled variants are evaluated separately.

29. **What is the uncertainty band?** The predeclared 0.35 to 0.65 score interval produces Not sure, with additional quality and unfamiliar-feature checks.

30. **Does the unfamiliar-feature gate guarantee safe unknown detection?** No. It is a heuristic based on train/validation feature distances and has not been validated as universal out-of-distribution detection.

31. **What was the actual test accuracy?** 60 percent among ten analyzed held-out videos; this is not a strong real-world reliability result.

32. **What is test coverage?** The fraction of reserved inputs receiving usable scores: 10 of 16, or 62.5 percent.

33. **What are false positives and false negatives?** Real videos flagged fake and fake videos missed as real, respectively. Both rates were 40 percent in the analyzed pilot test.

34. **Why can ROC-AUC look better than accuracy?** AUC measures ranking across thresholds, while accuracy uses one boundary. Good ranking does not guarantee correct individual decisions.

35. **How do PR-AUC and average precision differ?** They summarize precision and recall using different integration conventions; report them under distinct names.

36. **What do Brier score and ECE measure?** Probability-style score error and binned score-frequency disagreement; they can be unstable on a very small test.

37. **What does a bootstrap interval mean?** It estimates variability by resampling source groups, subject to independence assumptions and very limited group counts here.

38. **How was robustness measured?** Unchanged weights analyzed held-out sources and fixed compression, blur and darker-lighting derivatives.

39. **Are those derivatives independent external data?** No. They test picture changes on existing sources, not transfer to an unrelated dataset.

40. **Do passing software tests prove unknown-video accuracy?** No. They verify decoding, splits, shapes, inference contracts and UI behavior; classification quality requires representative labeled evaluation.

41. **Why test filename renaming?** An unchanged video should receive the same score under a different name; labels must not leak through filenames at inference.

42. **How is upload privacy handled?** Temporary disk video is deleted after analysis; preview and extracted evidence remain in session until cleared or disconnected.

43. **What makes the report safe to render?** User-controlled strings are HTML-escaped and the report does not embed raw uploaded media.

44. **Who performed the work?** The implementation is AI-assisted. Team members must record actual review, experimentation and understanding rather than invented authorship.

45. **What is the next scientific step?** Obtain more licensed independent subjects/methods, verify identity/source metadata, lock a fresh external evaluation set and document failures as carefully as successes.
