# Bundled face detector

The face detector is OpenCV's ResNet-10 SSD face detector, separate from the
EfficientNet classifier used to look for manipulation patterns.

- Weights: https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
- Definition: https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt
- Weight SHA-256: `510ffd2471bd81e3fcc88a5beb4eae4fb445ccf8333ebc54e7302b83f4158a76`
- Definition SHA-256: `dcd661dc48fc9de0a341db1f666a2164ea63a67265c7f779bc12d6b3f2fa67e9`

Both files were retrieved over verified HTTPS from the official OpenCV repositories.
The app checks their pinned hashes before loading. The upstream model README and
OpenCV license are retained alongside them. The weights were supplied by the
OpenCV third-party model branch; no separate weight license was present on that
branch. OpenCV's license file applies to the accompanying OpenCV material.

No face detector download is needed at application runtime. EfficientNet's
separate official ImageNet weights are cached by torchvision on first use.
