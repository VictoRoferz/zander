# Dataset card — NAL defect detection

_Generated 2026-08-06 · snapshot `v1_2026-07-27_yolo` · split `split_v1` (seed 20260806, K=3)_

Single class: **Nicht ausreichend Lot** (insufficient solder). Brücke/Fahne boxes are dropped to background by design (kept in Label Studio; recorded as `other_boxes` for eval FP tagging).

## Photos

| state | count |
|---|---|
| annotated_negative | 199 |
| annotated_positive | 99 |
| unannotated | 183 |
| quarantined (excluded, any state) | 12 |
| **usable for train/eval** | **298** |

## Per capture session (usable photos)

| session | photos | positives | NAL boxes | buckets |
|---|---|---|---|---|
| 2026-06-12 | 2 | 2 | 2 | fold0:2 |
| 2026-06-15 | 181 | 6 | 12 | fold0:45, fold1:41, fold2:47, test:48 |
| 2026-06-24 | 34 | 31 | 49 | fold1:28, test:6 |
| 2026-07-20 | 81 | 60 | 200 | fold0:26, fold2:40, test:15 |

## Split composition (usable photos)

| bucket | photos | positives | NAL boxes |
|---|---|---|---|
| fold0 | 73 | 26 | 75 |
| fold1 | 69 | 26 | 41 |
| fold2 | 87 | 26 | 96 |
| test | 69 | 21 | 51 |

## NAL box sizes (sqrt-area px at native 2616×1960)

| bucket | boxes |
|---|---|
| 0–16 px | 12 |
| 16–32 px | 180 |
| 32–64 px | 69 |
| 64–128 px | 0 |
| >128 px | 2 |

min 10 · median 28 · max 708 px (263 boxes). Boxes >300 px: 1 (tile-fragmentation caveat → IoS merge at stitch time).

## Tiles

- tiler `v1`: 640 px / stride 512, encoding `jpeg95`
- 5960 tiles from 298 photos; 367 positive tiles, 484 tile boxes
- train oversample K = 7 (target positive fraction 0.3)
- dataset_fingerprint `38e6165f6e8060d2…`

## Review list

- `0fef68e4e70449d787e01b8140034479`: LS choice says iO but a (Fahne) box exists; consistent negative after single-class remap — review in Label Studio when convenient.
