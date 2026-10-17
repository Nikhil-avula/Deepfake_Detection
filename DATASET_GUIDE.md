# Dataset access, provenance and labels

No dataset was downloaded. The user confirmed **no FakeAVCeleb access yet**. Its official
[repository](https://github.com/DASH-Lab/FakeAVCeleb) and
[download instructions](https://github.com/DASH-Lab/FakeAVCeleb/blob/main/dataset/README.md)
were read successfully. They require the linked request form and owner acceptance before
sending a download script. Code being MIT-licensed does not make the data unrestricted.
Do not redistribute media or run an unreviewed download script blindly.

## Candidate comparison

This is a selection assessment, not a claim of downloaded or inspected files. “Unknown”
means a requirement must be checked with the actual release and license. Size estimates
and precise counts are intentionally not invented. Storage must be measured from the
approved release inventory. Rows without a verified official source remain unverified.

| Dataset | Media / labels and sync fit | Identity/source and scale | Access, integrity, external role |
|---|---|---|---|
| FakeAVCeleb | Audio/video, separate real/fake audio/video categories reported by authors; best provisional multimodal fit | Authors report 500 real + 19,500 fake; actual release/metadata not inspected | Official README verified; approval required; owner checksum availability unknown; primary only after structure audit |
| FaceForensics++ | Visual manipulations; not an independent spoofed-voice benchmark | Source/derivative grouping required; compression variants must stay together | Official repository verified; request/license process; external visual test candidate |
| Celeb-DF v2 | Visual face swaps; audio must not inherit visual fake labels | Identity/source links need verification after access | Official repository verified; request form; preferred external visual test |
| Celeb-DF++ | Release identity and official source unverified here | Unknown | Do not confuse with v2; no download or benchmark commitment |
| DFDC | Audio/video corpus; overall fake label alone insufficient for audio supervision | Large; original-video links/identities need audit | Meta destination proxy-denied; license and authorized distribution must be reviewed |
| Google DFD | Visual actor-based manipulations, associated with FaceForensics release | Source actor/recording links required | Mentioned in verified FaceForensics documentation; check exact terms; visual external candidate |
| DeeperForensics-1.0 | Primarily visual face forgery and perturbations | Large; actor/source grouping essential | Official repository page verified; release/license/files not obtained; visual robustness candidate |
| WildDeepfake | In-the-wild visual face sequences; audio/sync suitability not established | Source provenance may limit strict group splits | Official project page verified; actual license and media structure pending; external visual only if compatible |
| ForgeryNet | Broad forgery tasks; separate audio labels not established | Large; identity/derivative metadata unknown | Candidate URL returned 404; official release not verified, not an approved acquisition source |
| KoDF | Korean video dataset, audio-spoof labels not established | Large; subject grouping needed | Candidate URL returned 404; verify authors' official release and terms before acquisition |
| DeepfakeTIMIT | Visual/audio task support must be checked per release | Small subject set; donor/target separation essential | Idiap official destination proxy-denied; license and actual task metadata pending |
| UADFV | Small visual benchmark; genuine audio labels not established | Identity grouping necessary; limited size | Candidate URL returned 404; official origin not confirmed here |
| ASVspoof | Audio-only spoof tasks; no video/lip-sync | Protocol-defined speaker/attack partitions | Official domain proxy-denied; choose challenge edition, protocol, terms and task before audio external evaluation |

Verified official pages on 2026-10-17: DASH-Lab/FakeAVCeleb, ondyari/FaceForensics,
yuezunli/celeb-deepfakeforensics, EndlessSora/DeeperForensics-1.0,
deepfakeinthewild/deepfake-in-the-wild. HTTP access verifies documentation, not dataset
permission, integrity or availability. Dataset-specific official adapters cannot be
certified before inspecting the approved metadata. The current importer deliberately
uses an explicit normalized contract rather than guessed directory names.

## Required action and acquisition

1. Complete the request form linked by the FakeAVCeleb owners. Review consent, use,
   storage and redistribution terms with your institution.
2. After approval, inspect the provided download instructions and obtain the release
   name/version, actual tree and metadata. Keep private URLs out of Git and chat logs.
3. If an official SHA-256 exists, `deepguard download HTTPS_URL data/raw/archive --sha256
   OFFICIAL_SHA256` supports resume and verification. It does not bypass authentication
   or unpack/execute content. If owners supply no checksum, use their authorized workflow
   and record locally computed hashes as change detection, **not proof of authenticity**.
4. Record license acceptance, approved source, release date/version and archive hash in
   a local provenance JSON. Do not fabricate this information from file names.
5. Build a normalized CSV from **actual verified metadata**, with absolute local paths:

```text
path,label,visual_label,audio_label,identities,sources,method,dataset
```

`label`: overall real=0/fake=1. `visual_label` and `audio_label`: genuine modality=0,
manipulated modality=1, genuinely unknown=-1. A fake voice with a real face is
`label=1,visual_label=0,audio_label=1`; never copy the overall label into both columns.
`identities`: pipe-separated, globally consistent donor **and** target subject IDs.
`sources`: pipe-separated IDs for **every** parent recording, including donor audio.
Do not prepend dataset names if the same subject appears across datasets. `method` is
the genuine manipulation category; `dataset` includes the actual release name.

## Inventory, quality, splits

```bash
deepguard inventory data/manifests/input.csv data/manifests/inventory.csv
deepguard split data/manifests/inventory.csv data/manifests/splits.csv
deepguard prepare data/manifests/splits.csv
# Once an authorized landmark task asset is present:
deepguard prepare data/manifests/splits.csv --landmark-asset checkpoints/face_landmarker.task
```

Inventory stores SHA-256, codec/FPS/duration/audio presence and probe failures, plus
dataset/class/method counts and manifest hash. FFprobe is a structural check; full
preprocessing adds decode checks but does not certify every packet. Investigate failures
and explicitly log exclusions. Split refuses known corrupt rows. Allocation is seeded
connected-component grouping, approximately 60/10/10/10/10 for train/val/fusion/
calibration/test with at least five components. All donor/target identities, all sources
and identical file hashes stay in one partition. Exact class balance is not guaranteed;
inspect each split and record any predeclared redesign. Do not keep reshuffling to
maximize test performance. Save split files and their hashes with the experiment.

For a constrained subset, select groups before splitting using documented class/method
strata; record excluded groups. No automatic sample selection claims are made here.
External datasets must be checked for subjects/source overlap with training data too.

## Landmark asset

The optional `mediapipe==0.10.32` library uses the official FaceLandmarker task model.
See <https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker> for model
terms and official model links. The attempted official
`storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`
download was proxy-denied (403). No substitute or unverified asset was installed.
Add `storage.googleapis.com` to the environment allowlist, review the model terms,
download through the official route and record its hash/version before using it.
“latest” is not a reproducible pin: retain the exact downloaded asset hash with runs.
The code will explicitly omit lip-sync evidence until this asset is present.

Dataset/domain access requirements: request forms use `docs.google.com` and
`sites.google.com`; ASVspoof uses `www.asvspoof.org`; DeepfakeTIMIT uses `www.idiap.ch`;
Meta uses `ai.meta.com`. Download hosts issued by owners are still unknown. Do not grant
wildcard Internet access just to make a failed dataset request pass.
