# Owner-provided video dataset

The current dataset is the owner's 98 uploaded videos: 49 labeled real (`0000.mp4` to
`0048.mp4`) and 49 labeled fake (`0000_fake.mp4` to `0048_fake.mp4`). These labels come
from the owner's naming instruction. Dataset origin/license, actual manipulation methods,
participant consent and donor identities have not been independently established; no
named benchmark provenance is claimed. The owner should confirm permitted research use.

All files passed structural video probing. Inventory records exact SHA-256, size,
duration, dimensions, frame rate and codec. A successful probe does not establish that
a label is correct or certify every packet. Face-quality checks exclude 25 inputs from
model fitting/evaluation; exclusions and reasons remain visible in reports/results.

Related original/fake filename stems stay together. Identical content is also grouped;
identical files carrying different labels cause a hard error. This source-pair grouping
is conservative but does not prove subjects/donors differ across splits. Supply verified
identity/source metadata for a stronger future experiment; do not invent subject IDs.

Fixed seed 42 allocates 29 source pairs to train, 6 to validation, 6 to calibration and
8 to final testing. Before face-quality exclusions this is 58/12/12/16 videos. After
exclusions it is 42/9/12/10. Both classes remain in each usable partition. No frame,
augmentation or derivative can move independently of its original source group.

The uploaded attachment index is local outside the repository. To reproduce ingestion
from an authorized local JSON index containing path/name/label records:

```bash
deepguard import-uploads /path/to/upload_index.json
```

Alternatively retain the supplied private manifest with columns path, file, label,
source_group, sha256 and split, preserving grouping and hashes. Import never infers
class labels from the visual appearance or generates additional fake examples. Keep raw
videos, face caches and private manifests out of Git. Hashes detect file changes, not
truthfulness or authenticity. A larger independently sourced dataset is needed to
support broad unknown-video conclusions.
