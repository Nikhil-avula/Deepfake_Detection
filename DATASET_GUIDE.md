# Owner-provided video dataset

The expanded inventory has 822 unique files: 319 Real and 503 Fake. Labels were supplied by the owner. Origin, license, identities and donor relationships were not independently verified. No named benchmark provenance is claimed.

The balanced candidate selection has 638 files: 319 per class, comprising 102 original files and 217 supplied augmentations per class. The remaining 184 fake files stay in reserve. Augmentations are variations, not independent sources.

Private manifests under data/manifests contain the complete inventory (expanded_inventory.csv), equal candidate selection (balanced_pending.csv), extra files (balance_reserve.csv), source-family assignments (splits.csv), and actual quality and selection outcomes (prepared.csv).

Content hashes deduplicate copies and detect changed files. Legacy 0000/0000_fake stems stay together. New vN/vsN files and real_vN/fake_vsN augmentations share a conservative expanded:N family. This groups possible relationships without claiming that matching numbers establish identity.

Historical split assignments are preserved. New families are assigned with seed 42. After quality filtering, training is balanced separately for original and augmented files; validation, calibration and testing use balanced original files only. Historical test originals are monitored separately from the primary new-source test cohort. Files reserved for balance or because they are evaluation augmentations are distinguished from quality failures. Balancing deletes no files.

To prepare and train from authorized local data:

    deepguard prepare-dataset data/manifests/balanced_pending.csv --previous-splits data/manifests/splits.csv
    deepguard train data/manifests/splits.csv --balanced

Identity-disjoint evaluation requires actual subject/donor metadata, which is unavailable. Raw files and private manifests are not committed to Git and are not needed for inference.
