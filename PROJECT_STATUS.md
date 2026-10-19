# Current status — V004 two-pass detector

Standard predictions are preserved. Insufficient face evidence triggers an additional
24-frame search and a separately trained retry classifier. Both trained JSON heads and
checksummed OpenCV SSD assets are included. Audio is not used.

638 balanced candidates: 616 quality eligible after retry, previously 593.
Standard training: 177/class; retry training: 179/class. All fitting partitions are balanced.
Reused reserved originals: 20/20 analyzed; 15/20 correct at 0.5; 14 clear answers.
Original scored outputs are unchanged. This is regression evidence, not unseen-video
accuracy. See EVALUATION_REPORT.md and component training reports for full evidence.

The owner authorized publishing this version to GitHub for fresh-clone testing. See INSTALLATION.md to run locally.
Current validation is recorded in experiments/validation.txt. Private training videos are
not required for new uploads. Generic ImageNet encoder weights download on first use
unless cached. Windows verification still requires the owner's computer.
