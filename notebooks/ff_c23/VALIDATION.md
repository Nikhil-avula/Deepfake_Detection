# Validation performed locally

The notebook contains 25 cells; all code cells compile. No FaceForensics++ videos
were downloaded here and no Kaggle GPU session was launched.

A CPU synthetic smoke test exercises strict folder/filename indexing, exclusion of
cross-partition pairs, official split ID disjointness, variable-length video batches,
classifier training, fine-tuning, checkpoint resume, validation threshold selection,
final metric generation and ZIP/app packaging. It uses a tiny substitute network;
its numerical results are not detection accuracy and must not be reported as such.
The export-packaging test stubs its predictor. See validate_notebook.py.

Separately, the actual EfficientNet-B0 runtime was checked with randomly initialized
weights and the user's previously supplied video: face extraction, full state-dict
serialization/loading and CPU-vs-cached-crop score equality passed. Blank-video
rejection, video-handle release and damaged model integrity rejection passed.
These checks establish functioning paths, not a trained detector's correctness.

Still requiring Kaggle execution: attached dataset layout/count verification,
network downloads, GPU mixed precision and memory capacity, actual training,
full-data metrics, export parity with trained weights, and local Windows execution.
No accuracy or wall-clock estimate has been inferred from the smoke tests.
