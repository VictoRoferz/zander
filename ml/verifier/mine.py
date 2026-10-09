"""
Out-of-fold candidate mining for the verifier (standard stacking CV).

    python verifier/cli.py mine --family yolo26n_def3 --run-name v1
    python verifier/cli.py mine --runs <F>_fold0 <F>_fold1 <F>_fold2 --run-name v1

Candidates are the detections each FOLD model made on ITS OWN held-out fold
(the fold eval runs' predictions.json): no candidate was ever seen in the
training of the detector that proposed it.

OOF guard (refuses unless --allow-non-oof, which stamps the run `smoke`):
  - one run per fold0/fold1/fold2 (never test/probe/train_always: always refused);
  - each run's model was trained without its bucket: the model's fold, read from
    predictions.source -> metrics.json -> the bundle's meta.json, equals the
    run's bucket;
  - three distinct model sha256s from one detector family.

Per photo the top_k detections by score become candidates, labelled
  1  same-class IoU >= 0.3 with a GT box (the scorer's match IoU),
 -1  best same-class IoU in [0.1, 0.3) — ambiguous localisation, ignored,
  0  otherwise; neg_type: wrong_class (IoU >= 0.3 with a GT box of another
     schema class), other_defect (IoU >= 0.1 with any other real defect,
     incl. out-of-schema ones), background.
Labels are IoU-based, not one-to-one: a duplicate detection of one defect is a
positive for the verifier (it is the IoS merge's job to remove duplicates).

A class is STACKED when it has >= min_pos_per_class positive detector
candidates over the three folds; otherwise it is PASSTHROUGH (keeps the
detector score; today Brücke/Fahne — board B is train_always, so they have no
out-of-fold data). For stacked classes every GT box is added once more as a
`gt` candidate: extra positives for VERIFIER training only (never a stacker
row, never a cascade candidate; the training jitter shifts it like any other).

Smoke mode (--allow-non-oof with guard violations): candidates.json carries
`smoke: true` and every downstream artifact inherits it (exit decision
invalid-smoke). With `candidates.smoke_inject_gt` the GT boxes are also
injected as `smoke_gt` pseudo-detections with random detector scores, so that
the stack path is exercised even behind a detector that finds nothing (the
smoke model does not). Never possible in a real run.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Optional

from config.settings import settings
from dataprep.ingest import load_manifest
from eval.run_eval import load_predictions
from eval.scorer import iou
from verifier.common import (
    CANDIDATES, FOLDS, Refusal, git_state, read_json, resolve_model_path, sha256_file, utc_now,
    vrun_dir, write_json_atomic,
)
from verifier.config import VerifierConfig, load_config

logger = logging.getLogger("ml.verifier.mine")

FORMAT_VERSION = 1
KIND = "zander-verifier-candidates"
_FOLD_SUFFIX = re.compile(r"_fold\d+$")


class OOFGuardError(Refusal):
    """The eval runs are not valid out-of-fold data for the verifier."""


# ------------------------------------------------------------------ runs


def family_runs(family: str) -> list[str]:
    """A family name resolves to its own-fold eval runs <F>_fold0..2."""
    return [f"{family}_{f}" for f in FOLDS]


def derive_family(run_name: str) -> str:
    return _FOLD_SUFFIX.sub("", run_name.split("__")[0])


def run_info(run: str, doc: Optional[dict] = None) -> dict:
    """Provenance of one eval run: bucket, the model's own fold, sha, family."""
    doc = doc if doc is not None else load_predictions(run)
    mpath = settings.eval_dir / run / "metrics.json"
    metrics = read_json(mpath) if mpath.exists() else {}
    src = doc.get("source") or {}
    msrc = metrics.get("source") or {}
    model_str = src.get("model_resolved") or msrc.get("model_resolved") or metrics.get("model") \
        or doc.get("model")
    model_path = resolve_model_path(model_str)
    meta_path = model_path.parent / "meta.json" if model_path else None
    meta = read_json(meta_path) if meta_path and meta_path.exists() else {}
    sha = src.get("model_sha256") or msrc.get("model_sha256") or metrics.get("model_sha256")
    if not sha and model_path and model_path.is_file():
        sha = sha256_file(model_path)
    return {
        "eval_run": run,
        "bucket": doc.get("bucket"),
        "kind": src.get("kind", "detector"),
        "fold": src.get("fold") or msrc.get("fold") or metrics.get("fold") or meta.get("fold"),
        "fold_source": ("predictions.source" if src.get("fold") else
                        "metrics.json" if (msrc.get("fold") or metrics.get("fold")) else
                        "meta.json" if meta.get("fold") else None),
        "model": str(model_str) if model_str else None,
        "model_resolved": str(model_path) if model_path else None,
        "model_sha256": sha,
        "family": src.get("family") or meta.get("family") or derive_family(run),
        "bundle_run": src.get("bundle_run") or meta.get("run_name"),
        "adapter": src.get("adapter") or metrics.get("adapter") or meta.get("adapter") or "ultralytics",
        "conf_floor": src.get("conf_floor") if src.get("conf_floor") is not None
        else (metrics.get("conf_floor") if metrics.get("conf_floor") is not None else 0.001),
        "schema": doc.get("schema"),
        "class_names": list(doc.get("class_names") or []),
        "threshold": metrics.get("threshold"),
        "mean_ms_per_photo": metrics.get("mean_ms_per_photo"),
        "tiler_version": metrics.get("tiler_version"),
        "split_id": metrics.get("split_id"),
        "snapshot_id": metrics.get("snapshot_id"),
        "dataset_fingerprint": metrics.get("dataset_fingerprint") or meta.get("dataset_fingerprint"),
        "n_photos": len(doc.get("photos") or []),
    }


def check_oof(infos: list[dict], candidate_floor: float = 0.001) -> list[str]:
    """Guard: hard errors raise OOFGuardError; returns the OVERRIDABLE
    violations (non-empty = not valid out-of-fold candidate data)."""
    hard = []
    buckets = [i["bucket"] for i in infos]
    if sorted(buckets) != sorted(FOLDS):
        hard.append(f"need exactly one eval run per {list(FOLDS)}, got buckets {buckets}")
    for i in infos:
        if i["kind"] != "detector":
            hard.append(f"{i['eval_run']}: source.kind {i['kind']!r} is not a detector run")
    if len({i["schema"] for i in infos}) != 1 or len({tuple(i["class_names"]) for i in infos}) != 1:
        hard.append("runs mix schemas/class names")
    if hard:
        raise OOFGuardError("; ".join(hard))
    soft = []
    for i in infos:
        if i["fold"] is None:
            soft.append(f"{i['eval_run']}: the model's training fold is unknown")
        elif i["fold"] != i["bucket"]:
            soft.append(
                f"{i['eval_run']}: model fold {i['fold']!r} ({i['fold_source']}) != bucket "
                f"{i['bucket']!r} — its predictions are not out-of-fold"
            )
    shas = [i["model_sha256"] for i in infos]
    if None in shas:
        soft.append("a model sha256 is unknown (model file missing?)")
    elif len(set(shas)) != len(shas):
        soft.append(f"the runs share model files (sha {sorted({s[:12] for s in shas})}) — "
                    "three distinct fold models are required")
    fams = sorted({i["family"] for i in infos})
    if len(fams) != 1:
        soft.append(f"the runs mix detector families {fams}")
    for i in infos:
        if i["conf_floor"] is not None and i["conf_floor"] > candidate_floor + 1e-12:
            soft.append(f"{i['eval_run']}: evaluated at conf floor {i['conf_floor']} > the "
                        f"candidate floor {candidate_floor} — low-score candidates are missing")
    return soft


# ------------------------------------------------------------------ labels


def top_k_indices(scores, k: int) -> list[int]:
    """Indices of the k highest scores (stable on ties: earlier index first)."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]


def label_candidate(box, c: int, gt, other_gt, iou_pos: float = 0.3,
                    ignore_lo: float = 0.1) -> tuple[int, Optional[str], float]:
    """(label, neg_type, best same-class IoU) — see the module doc."""
    best = max((iou(box, g) for k, g in gt if k == c), default=0.0)
    if best >= iou_pos:
        return 1, None, best
    if best >= ignore_lo:
        return -1, None, best
    other_cls = max((iou(box, g) for k, g in gt if k != c), default=0.0)
    if other_cls >= iou_pos:
        return 0, "wrong_class", best
    other_any = max([other_cls] + [iou(box, g) for _, g in other_gt])
    if other_any >= ignore_lo:
        return 0, "other_defect", best
    return 0, "background", best


def _subsample(photos: list[dict], fraction: float, seed: int) -> list[dict]:
    """Deterministic, stratified (positive / not) photo subset."""
    if fraction >= 1.0:
        return photos

    def key(p):
        return hashlib.sha256(f"{seed}:{p['capture_id']}".encode()).hexdigest()

    keep = set()
    for grp in ([p for p in photos if p["gt"]], [p for p in photos if not p["gt"]]):
        grp = sorted(grp, key=key)
        keep.update(p["capture_id"] for p in grp[: math.ceil(fraction * len(grp))])
    return [p for p in photos if p["capture_id"] in keep]


# ------------------------------------------------------------------ mine


def mine(
    runs: Optional[list[str]] = None,
    family: Optional[str] = None,
    allow_non_oof: bool = False,
    vrun: str = "v1",
    cfg: Optional[VerifierConfig] = None,
    fraction: float = 1.0,
) -> Path:
    cfg = cfg or load_config()
    cc = cfg.candidates
    if bool(runs) == bool(family):
        raise ValueError("give either the three fold eval runs or a family name")
    runs = list(runs) if runs else family_runs(family)
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1]")

    docs = {r: load_predictions(r) for r in runs}
    infos = [run_info(r, docs[r]) for r in runs]
    violations = check_oof(infos, cc.candidate_floor)
    if violations and not allow_non_oof:
        raise OOFGuardError(
            "not out-of-fold — refusing to mine (use --allow-non-oof for a SMOKE run): "
            + "; ".join(violations)
        )
    smoke = bool(violations)
    if smoke:
        logger.warning("SMOKE run (--allow-non-oof): " + "; ".join(violations))
    infos.sort(key=lambda i: i["bucket"])
    class_names = infos[0]["class_names"]
    schema = infos[0]["schema"]
    fam = infos[0]["family"] if len({i["family"] for i in infos}) == 1 else "+".join(
        sorted({i["family"] for i in infos}))

    manifest = {e["capture_id"]: e for e in load_manifest()["entries"]}
    rng = random.Random(cfg.train.seed)
    photos_out, cands = [], []
    for info in infos:
        doc = docs[info["eval_run"]]
        chosen = _subsample(doc["photos"], fraction, cfg.train.seed)
        for p in chosen:
            cid = p["capture_id"]
            e = manifest.get(cid, {})
            width, height = p.get("width") or e.get("width"), p.get("height") or e.get("height")
            if not width or not height:
                raise ValueError(f"no image size for {cid} (not in predictions nor manifest)")
            scores = list(p["scores"])
            photos_out.append({
                "capture_id": cid, "fold": info["bucket"], "eval_run": info["eval_run"],
                "group": p["group"], "board": p["board"], "pose": p["pose"],
                "gt_verdict": p["gt_verdict"], "width": int(width), "height": int(height),
                "image_path": e.get("image_path"),
                "gt": [[c, list(b)] for c, b in p["gt"]],
                "other_gt": [[n, list(b)] for n, b in p["other_gt"]],
                "det_photo_score": max(scores, default=0.0), "n_det": len(scores),
            })
            for rank, i in enumerate(top_k_indices(scores, cc.top_k)):
                box, c = list(p["boxes"][i]), int(p["labels"][i])
                label, neg, best = label_candidate(
                    box, c, p["gt"], p["other_gt"], cc.iou_positive, cc.ignore_band[0])
                cands.append({"capture_id": cid, "fold": info["bucket"], "kind": "det",
                              "rank": rank, "class": c, "box": box, "det_score": scores[i],
                              "label": label, "neg_type": neg, "iou": round(best, 4)})
            if smoke and cc.smoke_inject_gt:
                top = max(scores, default=0.0)
                for c, b in p["gt"]:
                    s = rng.uniform(cc.candidate_floor, max(top, 2 * cc.candidate_floor))
                    cands.append({"capture_id": cid, "fold": info["bucket"], "kind": "smoke_gt",
                                  "rank": None, "class": int(c), "box": list(b),
                                  "det_score": s, "label": 1, "neg_type": None, "iou": 1.0})

    positives = Counter(c["class"] for c in cands if c["label"] == 1 and c["kind"] != "gt")
    stack = [i for i in range(len(class_names)) if positives[i] >= cc.min_pos_per_class]
    passthrough = [i for i in range(len(class_names)) if i not in stack]
    gt_pos = Counter()
    for ph in photos_out:
        for c, b in ph["gt"]:
            if c in stack:
                gt_pos[c] += 1
                cands.append({"capture_id": ph["capture_id"], "fold": ph["fold"], "kind": "gt",
                              "rank": None, "class": int(c), "box": list(b), "det_score": None,
                              "label": 1, "neg_type": None, "iou": 1.0})
    for n, c in enumerate(cands):
        c["id"] = n

    per_fold = {f: Counter() for f in FOLDS}
    for c in cands:
        if c["kind"] != "gt":
            per_fold[c["fold"]][(class_names[c["class"]], c["label"])] += 1
    doc = {
        "format_version": FORMAT_VERSION, "kind": KIND, "vrun": vrun, "created": utc_now(),
        "schema": schema, "class_names": class_names, "family": fam,
        "runs": infos, "smoke": smoke, "oof_violations": violations,
        "fraction": fraction, "config": cfg.to_dict(), "config_sha256": cfg.sha256, **git_state(),
        "stack_classes": stack, "passthrough_classes": passthrough,
        "positives_per_class": {class_names[i]: positives[i] for i in range(len(class_names))},
        "gt_positives_per_class": {class_names[i]: gt_pos[i] for i in stack},
        "counts_per_fold": {f: {f"{k[0]}|{k[1]}": v for k, v in sorted(cnt.items())}
                            for f, cnt in per_fold.items()},
        "photos": photos_out, "candidates": cands,
    }
    if not stack:
        logger.warning(
            "no class reaches min_pos_per_class=%d positive candidates %s — every class is "
            "passthrough, the verifier has nothing to learn", cc.min_pos_per_class,
            {class_names[i]: positives[i] for i in range(len(class_names))})
    out = vrun_dir(vrun, create=True) / CANDIDATES
    sha = write_json_atomic(out, doc, indent=None)
    logger.info(
        f"mined {len(cands)} candidates on {len(photos_out)} photos "
        f"(stack {[class_names[i] for i in stack]}, passthrough "
        f"{[class_names[i] for i in passthrough]}, positive candidates "
        f"{ {class_names[i]: positives[i] for i in range(len(class_names))} }) -> {out} "
        f"[sha {sha[:12]}]"
    )
    return out


def load_candidates(vrun: str) -> tuple[dict, str]:
    path = vrun_dir(vrun) / CANDIDATES
    if not path.exists():
        raise FileNotFoundError(f"run `verifier/cli.py mine --run-name {vrun}` first — missing {path}")
    data = path.read_bytes()
    return json.loads(data.decode("utf-8")), hashlib.sha256(data).hexdigest()
