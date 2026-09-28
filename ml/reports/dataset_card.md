# Dataset card — solder-joint defect detection (zander)

_Generated 2026-09-28 · snapshot `v2_2026-08-17_yolo` · split `split_v2` (frozen at 2026-08-17, K=3) · views `ds_v2_def3`_

Required output per photo: **N.iO + detected defect class(es)**, or **i.O**. Box classes: Nicht ausreichend Lot (NAL), Brücke, Fahne.

## Photos

| state | photos |
|---|---|
| annotated | 329 |
| review | 2 |
| unannotated | 235 |
| quarantined (no usable PCB / fallback; any state) | 87 |
| **usable for train/eval** | **327** (iO 194, NiO 133) |

## Class ↔ board confound (usable photos)

| class | board A (boxes / photos) | A2 | B |
|---|---|---|---|
| NAL | 191 / 87 | 70 / 10 | 0 / 0 |
| Brücke | 2 / 2 | 3 / 2 | 516 / 32 |
| Fahne | 11 / 9 | 0 / 0 | 179 / 25 |
| iO photos | 23 seated · 170 handheld | 0 seated · 0 handheld | 1 seated · 0 handheld |

- **Brücke/Fahne live almost only on board B — one panel, one 16-min session.** Board B is `train_always` (never validated): its Brücke/Fahne numbers can only be reported as within-panel memorisation.
- **Honest cross-product check** = the probe: the board-A/A2 Brücke/Fahne boxes are held out of every training view (tiles touching them are dropped). 16 boxes: Brücke 5, Fahne 11 — results are hit counts, not rates.
- **Production-pose negatives are scarce**: the false-call rate is measured on *seated* iO photos only; handheld iO photos are training negatives (capped) and a separate row.

## Split composition (usable photos)

| bucket | photos | NiO | iO seated | iO handheld | NAL boxes | Brücke boxes | Fahne boxes | groups |
|---|---|---|---|---|---|---|---|---|
| fold0 | 71 | 27 | 2 | 42 | 72 | 1 | 3 | 10 |
| fold1 | 73 | 27 | 0 | 46 | 97 | 4 | 2 | 15 |
| fold2 | 75 | 27 | 10 | 38 | 56 | 0 | 5 | 16 |
| train_always | 33 | 32 | 1 | 0 | 0 | 516 | 179 | 1 |
| test | 75 | 20 | 11 | 44 | 36 | 0 | 1 | 10 |

Groups (`groups_v2.json`, edges {'time': 445, 'time_cut_board_change': 39, 'reshot': 10, 'board_b': 6}): 42 contain NAL; largest [21, 10, 7, 6, 4] NAL photos — the effective sample size is the number of groups, not photos.

## Box sizes (sqrt-area px at native 2616×1960, usable photos)

| class | boxes | median | 0–16 | 16–32 | 32–64 | 64–128 | >128 |
|---|---|---|---|---|---|---|---|
| NAL | 261 | 28 | 12 | 180 | 69 | 0 | 0 |
| Brücke | 521 | 34 | 5 | 182 | 320 | 14 | 0 |
| Fahne | 190 | 39 | 3 | 46 | 139 | 2 | 0 |

## Colour / exposure per session vs. augmentation ranges

| session | board | photos | luma | board L* | a* | b* | hue(a*b*) |
|---|---|---|---|---|---|---|---|
| S05 2026-06-15 | A | 179 | 121 | 55 | -32.0 | -2.2 | -176° |
| S06 2026-06-24 | A | 34 | 177 | 75 | -20.3 | -1.7 | -175° |
| S07 2026-07-20 | A | 71 | 140 | 57 | -20.1 | -2.3 | -173° |
| S07 2026-07-20 | A2 | 10 | 138 | 56 | -26.0 | 2.5 | 175° |
| S09 2026-08-17 | B | 33 | 162 | 74 | -16.2 | -1.6 | -174° |

Session luma spans 121–177 (±20% around the median) → online `hsv_v=0.35`, `hsv_s=0.4`, small `hsv_h=0.015` (solder colour is signal). The large board-A ↔ board-B colour shift is NOT covered by HSV jitter — the offline cross-board colour transfer handles it.

## Training views `ds_v2_def3`

- tiler `v2` 640 px / stride 512, `jpeg95`; 6540 base tiles from 327 photos
- offline augmentation: `{'d4': True, 'ct_p': 0.3, 'handheld_cap': True, 'rfs': True, 'pos_share_target': 0.3, 'r_max': 8, 'fixed_k': 7, 'partial_min_visibility': 0.5}`
- dataset_fingerprint `2c3b0cde1b8ec1c9…`

| view | entries | files | pos share | instances | drop partial | drop probe | drop handheld | D4 parity Δ | CT pos/neg |
|---|---|---|---|---|---|---|---|---|---|
| fold0_train | 3338 | 3338 | 0.30 | Brücke 1049 · Fahne 689 · NAL 471 | 75 | 22 | 633 | 0.020 | 0.30/0.30 |
| fold0_val | 1420 | 1420 | 0.09 | Brücke 4 · Fahne 3 · NAL 161 | – | – | – | 0.062 | 0.00/0.00 |
| fold1_train | 3469 | 3469 | 0.30 | Brücke 1054 · Fahne 699 · NAL 478 | 69 | 13 | 490 | 0.027 | 0.31/0.31 |
| fold1_val | 1460 | 1460 | 0.08 | Brücke 11 · Fahne 5 · NAL 158 | – | – | – | 0.066 | 0.00/0.00 |
| fold2_train | 3031 | 3031 | 0.30 | Brücke 893 · Fahne 613 · NAL 447 | 73 | 23 | 718 | 0.021 | 0.31/0.30 |
| fold2_val | 1500 | 1500 | 0.05 | Fahne 10 · NAL 99 | – | – | – | 0.127 | 0.00/0.00 |
| dev_all_train | 4606 | 4606 | 0.30 | Brücke 1280 · Fahne 884 · NAL 739 | 80 | 29 | 1030 | 0.017 | 0.30/0.30 |
| test | 1500 | 1500 | 0.03 | Fahne 4 · NAL 58 | – | – | – | 0.066 | 0.00/0.00 |

D4 parity Δ = max difference of the orientation histograms of positive vs negative entries (≈0 means orientation carries no label information).

## Review list

- `0fef68e4e70449d787e01b8140034479` (review): LS image choice says iO (June-24 export) but a Fahne box exists
- `38d0b5c48e8146bcb9d967db4715bc20` (review): LS image choice 'Weitere Überprüfung notwendig' but no box drawn
- `b6eeb74c0033498789957795df9e52fb` (annotated): one Fahne box (75x65 px, board A) looks more like a scratch — verify in LS
