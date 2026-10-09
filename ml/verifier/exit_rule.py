"""
The pre-registered exit rule (verifier/PREREGISTRATION.md) -> exit_report.{md,json}
and the `exit` block of stacker.json.

    python verifier/cli.py exit --run-name v1

(a) PRIMARY — pooled over the three held-out folds, NAL recall at 0.5 FPPI
    (IoU 0.3), cascade minus detector, paired board-group cluster bootstrap
    (the scorer primitives compare.py uses); passes iff the lower 95 % bound > 0.
(b) SECONDARY — cross-fitted thresholds for both systems on every held-out
    fold k: the detector at t(k) (choose-op rule on the two other folds), the
    cascade at tau(k) (chosen on the same two folds at t(k)). Counts NAL
    false-alarm boxes (>= threshold, unmatched at IoU 0.3), false-call photos
    (seated / handheld iO), escapes, and photos the cascade ADDS as escapes.
    Passes iff pooled NAL false alarms <= (1 - 0.30) x the detector's and zero
    added photo escapes (compared exactly on integers: 10 x cascade <= 7 x
    detector, so exactly 30 % fewer passes).
(c) LATENCY GATE — the detector (fold-0 model, conf floor 0.001) + the
    exported verifier on N photos with ONNX Runtime CPU: verifier p50 for 50
    candidates <= 200 ms and detector + verifier p50 <= 1.0 s.
(d) candidate ceiling — share of NAL GT boxes matched by a top-k candidate
    (and by any detection) on the held-out folds: what stage 2 can reach at all.
(e) inner-vs-outer verifier NAL AUROC (the nested-stacking train/apply gap).

Decision: ship iff (c) and ((a) or (b)). invalid-smoke (no decision) when the
run is not the pre-registered one: smoke / non-OOF data, a photo fraction < 1,
a config that is not the committed file byte for byte (overrides, another
path, an edited file: PREREGISTERED_CONFIG_SHA256) at mine, crossfit OR exit
time, uncommitted code under ml/ at any of the three steps, an exit with
--threads != 0 (ORT default threads) or another --latency-photos count, or a
random-init backbone in ANY of the seven cross-fit models. The gates are read
from the exit-time config, so these checks are what keeps `exit --set
exit.<gate>=...` from turning no-ship into ship. An earlier exit report is
archived to exit_archive/<UTC stamp>/, never overwritten (every exit is
reported), and listed in the new one. (d) and (e) are descriptive.
"""
from __future__ import annotations

import logging
import os
import platform
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Optional

import numpy as np

from dataprep.ingest import photo_path
from eval.photo_scorer import class_records, match_indices
from eval.run_eval import load_predictions
from eval.scorer import (
    FPPI_OPERATING, IOU_OPERATING, _froc_from_stats, _photo_stats, cluster_resample, iou,
    recall_at_fppi,
)
from infer.pipeline import TiledDetector, load_photo_rgb
from infer.verifier import Verifier, anchor_map, candidate_features
from verifier.common import (
    EXIT_ARCHIVE, EXIT_JSON, EXIT_MD, OOF_SCORES, STACKER, FOLDS, TRAIN_LOG, git_state, read_json,
    sha256_file, utc_now, vrun_dir, write_json_atomic, write_bytes_atomic,
)
from verifier.config import VerifierConfig, preregistration_problems
from verifier.crossfit import NAL, cascade_photos
from verifier.mine import load_candidates

logger = logging.getLogger("ml.verifier.exit")

PREREG = Path(__file__).resolve().parent / "PREREGISTRATION.md"
PRETRAINED = "imagenet1k_v1"  # the only backbone source a decision run may use


# ------------------------------------------------------------------ (a)


def _ci(xs: list[float]) -> Optional[list[float]]:
    if not xs:
        return None
    xs = sorted(xs)
    return [round(xs[int(0.025 * len(xs))], 4), round(xs[int(0.975 * len(xs)) - 1], 4)]


def paired_recall(det_photos: list[dict], casc_photos: list[dict], c: int,
                  n_boot: int = 1000, seed: int = 0) -> dict:
    """Recall @ 0.5 FPPI (IoU 0.3) of class c, cascade - detector, on the same
    photos; CI from a paired board-group cluster bootstrap."""
    if [p["capture_id"] for p in det_photos] != [p["capture_id"] for p in casc_photos]:
        raise ValueError("paired comparison needs the same photos in the same order")
    sd = _photo_stats(class_records(det_photos, c), IOU_OPERATING)
    sc = _photo_stats(class_records(casc_photos, c), IOU_OPERATING)
    clusters = [p["group"] for p in det_photos]

    def rec(stats, idx):
        pts, n_gt = _froc_from_stats([stats[i] for i in idx])
        return recall_at_fppi(pts, FPPI_OPERATING)[0] if n_gt else None

    full = range(len(det_photos))
    r_det, r_casc = rec(sd, full), rec(sc, full)
    if r_det is None:
        return {"n_gt": 0, "passed": False, "note": "no GT boxes of this class"}
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        idx = cluster_resample(len(det_photos), clusters, rng)
        a, b = rec(sc, idx), rec(sd, idx)
        if a is not None and b is not None:
            diffs.append(a - b)
    ci = _ci(diffs)
    return {
        "n_photos": len(det_photos), "n_groups": len(set(clusters)),
        "n_gt": sum(len(s[0]) for s in sd),
        "recall_detector": round(r_det, 4), "recall_cascade": round(r_casc, 4),
        "diff": round(r_casc - r_det, 4), "diff_ci95": ci, "bootstrap_n": len(diffs),
        "ci_unit": "board group", "passed": bool(ci and ci[0] > 0),
    }


# ------------------------------------------------------------------ (b)


def _fa_boxes(boxes, scores, gt) -> int:
    """Flagged boxes not matched to a GT box (greedy, IoU 0.3)."""
    if not boxes:
        return 0
    _, matched = match_indices(boxes, scores, gt, IOU_OPERATING)
    return sum(1 for m in matched if not m)


def crossfit_thresholds(cand: dict, oof: dict, det_by_id: dict, nal: int,
                        fa_reduction: float) -> dict:
    stack = set(oof["stack_classes"])
    rows = {r["id"]: r for r in oof["rows"]}
    per_photo = defaultdict(list)
    for c in cand["candidates"]:
        if c["kind"] != "gt":
            per_photo[c["capture_id"]].append(c)
    names = cand["class_names"]
    tot = defaultdict(int)
    per_fold = {}
    added_ids, rescued_ids = [], []
    for k in FOLDS:
        f = oof["folds"][k]
        t_k, tau_k = f["t_det"], f["tau"]
        hi = {names.index(n): h for n, h in f["hi"].items()}
        cnt = defaultdict(int)
        for ph in cand["photos"]:
            if ph["fold"] != k:
                continue
            cid = ph["capture_id"]
            dp = det_by_id[cid]
            gt_nal = [b for c, b in ph["gt"] if c == nal]
            positive, negative = bool(ph["gt"]), ph["gt_verdict"] == "iO"
            # detector at t(k): every detection of the eval run
            d_idx = [i for i, (s, lab) in enumerate(zip(dp["scores"], dp["labels"]))
                     if s >= t_k and lab == nal]
            cnt["det_fa"] += _fa_boxes([dp["boxes"][i] for i in d_idx],
                                       [dp["scores"][i] for i in d_idx], gt_nal)
            det_flag = max(dp["scores"], default=0.0) >= t_k
            # cascade at tau(k) / t(k): the top-k candidates
            c_boxes, c_scores, casc_flag = [], [], False
            for c in per_photo.get(cid, []):
                if c["class"] in stack:
                    p = rows[c["id"]]["p_outer"]
                    h = hi.get(c["class"])
                    flagged = p >= tau_k or (h is not None and c["det_score"] >= h
                                             and c["det_score"] >= t_k)
                    if flagged and c["class"] == nal:
                        c_boxes.append(tuple(c["box"]))
                        a = float(anchor_map(p, tau_k, t_k))
                        c_scores.append(max(a, c["det_score"]) if (h is not None and c["det_score"] >= h) else a)
                else:
                    flagged = c["det_score"] >= t_k
                casc_flag = casc_flag or flagged
            cnt["casc_fa"] += _fa_boxes(c_boxes, c_scores, gt_nal)
            if positive:
                cnt["positives"] += 1
                cnt["det_escapes"] += not det_flag
                cnt["casc_escapes"] += not casc_flag
                if det_flag and not casc_flag:
                    cnt["added_escapes"] += 1
                    added_ids.append(cid)
                if casc_flag and not det_flag:
                    cnt["rescued"] += 1
                    rescued_ids.append(cid)
            if negative:
                pose = ph["pose"] if ph["pose"] in ("seated", "handheld") else "other"
                cnt[f"negatives_{pose}"] += 1
                cnt[f"det_false_calls_{pose}"] += det_flag
                cnt[f"casc_false_calls_{pose}"] += casc_flag
        per_fold[k] = {"t_det": t_k, "tau": tau_k, **dict(cnt)}
        for key, v in cnt.items():
            tot[key] += v
    det_fa, casc_fa = tot["det_fa"], tot["casc_fa"]
    reduction = (det_fa - casc_fa) / det_fa if det_fa else None
    # exact: 1 - Fraction("0.3") = 7/10, so 63 <= 7/10 * 90 holds (in floats
    # (1 - 0.30) * 90 = 62.999... and exactly 30 % fewer false alarms failed)
    keep = 1 - Fraction(str(fa_reduction))
    passed = bool(det_fa > 0 and casc_fa <= keep * det_fa and tot["added_escapes"] == 0)
    return {"per_fold": per_fold, "pooled": dict(tot), "nal_fa_reduction": (
        None if reduction is None else round(reduction, 4)), "required_reduction": fa_reduction,
        "added_escape_photos": added_ids, "rescued_photos": rescued_ids, "passed": passed}


# ------------------------------------------------------------------ (c)


def latency(cand: dict, stacker_path: Path, n_photos: int, warmup: int, threads: int = 0,
            n_candidates: int = 50) -> dict:
    # the pre-registered bench model: the fold-0 detector
    info = next((r for r in cand["runs"] if r.get("bucket") == "fold0"), cand["runs"][0])
    model = Path(info["model_resolved"])
    ver = Verifier(stacker_path, threads=threads)
    det = TiledDetector(model, info["adapter"], conf_floor=ver.stacker.candidate_floor,
                        threads=threads)
    photos = sorted(cand["photos"], key=lambda p: p["capture_id"])
    if not photos:
        raise RuntimeError("no photos for the latency bench")
    stacked = sorted(ver.stacker.stacked) or [0]
    det_ms, ver_ms, ver50_ms, tot50, native_ms = [], [], [], [], []
    seq = [photos[i % len(photos)] for i in range(warmup + n_photos)]
    for n, ph in enumerate(seq):
        rgb = load_photo_rgb(photo_path({"image_path": ph["image_path"]}))
        d = det.detect(rgb)
        t0 = time.perf_counter()
        ver.rescore(rgb, d)
        t_actual = (time.perf_counter() - t0) * 1000.0
        # worst case: exactly n_candidates candidates through crops + ONNX + stacker
        order = np.argsort(-np.asarray(d.scores), kind="stable")[:n_candidates]
        boxes = [d.boxes[i] for i in order] or [(0.0, 0.0, 30.0, 30.0)]
        boxes = [boxes[i % len(boxes)] for i in range(n_candidates)]
        dets = [d.scores[i] for i in order] or [0.5]
        dets = [dets[i % len(dets)] for i in range(n_candidates)]
        t1 = time.perf_counter()
        lg = ver.logits(ver.windows(rgb, boxes))
        c = stacked[0]
        x = candidate_features(dets, lg[:, c], boxes, rgb.shape[1], rgb.shape[0])
        if ver.stacker.is_stacked(c):
            ver.stacker.final(c, ver.stacker.proba(c, x), dets)
        t50 = (time.perf_counter() - t1) * 1000.0
        # reference: 50 NAL-sized (30 px, s = 1) boxes -> the network cost alone
        h, w = rgb.shape[:2]
        ref = [(x, y, x + 30.0, y + 30.0) for x, y in zip(
            np.linspace(40, w - 70, n_candidates), np.linspace(40, h - 70, n_candidates))]
        t2 = time.perf_counter()
        lg = ver.logits(ver.windows(rgb, ref))
        x = candidate_features([0.5] * n_candidates, lg[:, c], ref, w, h)
        if ver.stacker.is_stacked(c):
            ver.stacker.final(c, ver.stacker.proba(c, x), [0.5] * n_candidates)
        t_native = (time.perf_counter() - t2) * 1000.0
        if n < warmup:
            continue
        native_ms.append(t_native)
        det_ms.append(d.elapsed_ms)
        ver_ms.append(t_actual)
        ver50_ms.append(t50)
        tot50.append(d.elapsed_ms + t50)

    def q(xs, p):
        return round(float(np.percentile(xs, p)), 1)

    out = {
        "host": platform.node(),
        "n_photos": len(det_ms), "warmup": warmup, "threads": threads or "ort-default",
        "detector_model": str(model), "n_candidates": n_candidates,
        "detector_ms": {"p50": q(det_ms, 50), "p95": q(det_ms, 95)},
        "verifier_actual_ms": {"p50": q(ver_ms, 50), "p95": q(ver_ms, 95)},
        "verifier_50_ms": {"p50": q(ver50_ms, 50), "p95": q(ver50_ms, 95)},
        "total_50_ms": {"p50": q(tot50, 50), "p95": q(tot50, 95)},
        "verifier_50_native_ms": {"p50": q(native_ms, 50), "p95": q(native_ms, 95)},
        "note": ("the gate uses each photo's actual top-50 detector candidates (repeated if "
                 "fewer); the native row times 50 NAL-sized boxes (s = 1) for reference: a "
                 "window for a huge box (s >> 1) costs 10-40 ms on a 2616x1960 photo, so a "
                 "detector that proposes many of them dominates the verifier time"),
    }
    return out


# ------------------------------------------------------------------ (d)


def candidate_ceiling(cand: dict, det_by_id: dict, nal: int) -> dict:
    per_photo = defaultdict(list)
    for c in cand["candidates"]:
        if c["kind"] == "det" and c["class"] == nal:
            per_photo[c["capture_id"]].append(c["box"])
    n_gt = hit_topk = hit_all = 0
    for ph in cand["photos"]:
        gts = [b for c, b in ph["gt"] if c == nal]
        dp = det_by_id[ph["capture_id"]]
        all_nal = [b for b, lab in zip(dp["boxes"], dp["labels"]) if lab == nal]
        for g in gts:
            n_gt += 1
            hit_topk += any(iou(b, g) >= IOU_OPERATING for b in per_photo.get(ph["capture_id"], []))
            hit_all += any(iou(b, g) >= IOU_OPERATING for b in all_nal)
    return {"n_gt": n_gt,
            "topk_share": round(hit_topk / n_gt, 4) if n_gt else None,
            "all_detections_share": round(hit_all / n_gt, 4) if n_gt else None,
            "top_k": cand["config"]["candidates"]["top_k"]}


# ------------------------------------------------------------------ decision


def exit_run_info(cfg: VerifierConfig, threads: int, latency_photos: Optional[int]) -> dict:
    """What the exit step itself ran with (recorded in the report, checked by
    invalid_reasons): its config, ORT threads, latency photo count, code state."""
    return {
        "config": {"name": cfg.name, "path": cfg.path, "sha256": cfg.sha256,
                   "overrides": dict(cfg.overrides)},
        "threads": int(threads or 0),
        "latency_photos": int(latency_photos or cfg.exit.latency_photos),
        "latency_photos_default": cfg.exit.latency_photos,
        **git_state(),
    }


def invalid_reasons(cand: dict, oof: dict, train_log: Optional[dict], exit_run: dict) -> list[str]:
    """Why this run is not the pre-registered one (empty = it is); any reason
    makes the decision invalid-smoke."""
    reasons = []
    if cand["smoke"]:
        reasons.append("smoke: candidates are not out-of-fold (" + "; ".join(cand["oof_violations"]) + ")")
    n_inj = sum(1 for c in cand["candidates"] if c["kind"] == "smoke_gt")
    if n_inj:
        reasons.append(f"{n_inj} GT boxes were injected as smoke_gt pseudo-detections: the "
                       "cascade 'finds' them by construction, the detector never proposed them")
    if cand["fraction"] < 1.0:
        reasons.append(f"only a {cand['fraction']:.0%} photo sample was mined")
    steps = (("mine", cand["config"], cand), ("crossfit", oof["config"], oof),
             ("exit", exit_run["config"], exit_run))
    for step, c, doc in steps:
        for problem in preregistration_problems(c):
            reasons.append(f"{step}: {problem} (not the pre-registered run)")
        dirty = doc.get("git_dirty")
        if dirty is None:
            reasons.append(f"{step}: code state unknown (no git_dirty recorded — not a git checkout?)")
        elif dirty:
            reasons.append(f"{step}: ran with uncommitted changes under ml/ (git "
                           f"{str(doc.get('git_sha'))[:12]} + local edits)")
    if exit_run["threads"] != 0:
        reasons.append(f"exit: --threads {exit_run['threads']} (the pre-registered latency gate uses "
                       "ONNX Runtime's default intra-op threads, --threads 0)")
    if exit_run["latency_photos"] != exit_run["latency_photos_default"]:
        reasons.append(f"exit: --latency-photos {exit_run['latency_photos']} (pre-registered: "
                       f"{exit_run['latency_photos_default']} photos)")
    if oof.get("weights_source") != PRETRAINED:
        reasons.append(f"V_final backbone not ImageNet-pretrained ({oof.get('weights_source')})")
    planned = sorted(((oof.get("plan") or {}).get("models")) or {})
    if train_log is None:
        reasons.append(f"{TRAIN_LOG} missing: the backbone source of the cross-fit models is unverifiable")
    else:
        logged = {m.get("name"): m.get("weights_source") for m in train_log.get("models") or []}
        missing = [n for n in planned if n not in logged]
        if missing:
            reasons.append(f"{TRAIN_LOG} lacks the cross-fit models {missing}")
        bad = {n: w for n, w in sorted(logged.items()) if w != PRETRAINED}
        if bad:
            reasons.append("backbone not ImageNet-pretrained in " + "; ".join(
                f"{n} ({w})" for n, w in bad.items()))
    return reasons


def decide(reasons: list[str], latency_passed: bool, primary_passed: bool,
           secondary_passed: bool) -> str:
    """PREREGISTRATION.md "Decision rule"."""
    if reasons:
        return "invalid-smoke"
    return "ship" if latency_passed and (primary_passed or secondary_passed) else "no-ship"


def archive_reports(d: Path) -> list[dict]:
    """Move the current exit_report.{json,md} (if any) to
    exit_archive/<UTC stamp>/ — an exit report is never overwritten — and
    list every archived report (oldest first)."""
    current = [p for p in (d / EXIT_JSON, d / EXIT_MD) if p.exists()]
    root = d / EXIT_ARCHIVE
    if current:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest, n = root / stamp, 1
        while dest.exists():
            dest, n = root / f"{stamp}_{n}", n + 1
        dest.mkdir(parents=True)
        for p in current:
            os.replace(p, dest / p.name)
        logger.warning(f"the earlier exit report was archived to {dest} (every exit is reported)")
    out = []
    for sub in sorted(root.iterdir()) if root.is_dir() else []:
        try:
            doc = read_json(sub / EXIT_JSON) if (sub / EXIT_JSON).is_file() else {}
        except ValueError:
            doc = {}
        out.append({"archive": f"{EXIT_ARCHIVE}/{sub.name}", "created": doc.get("created"),
                    "decision": doc.get("decision")})
    return out


def run(vrun: str, cfg=None, threads: int = 0, latency_photos: Optional[int] = None) -> dict:
    """The gates are read from `cfg` (default: the pre-registered file); a cfg
    that is not that file, --threads != 0 or another --latency-photos count
    turns the decision into invalid-smoke (invalid_reasons)."""
    from verifier.config import load_config

    cfg = cfg or load_config()
    ec = cfg.exit
    exit_run = exit_run_info(cfg, threads, latency_photos)
    d = vrun_dir(vrun)
    cand, cand_sha = load_candidates(vrun)
    oof = read_json(d / OOF_SCORES)
    if oof["candidates_sha256"] != cand_sha:
        raise RuntimeError("oof_scores.json does not belong to this candidates.json")
    train_log = read_json(d / TRAIN_LOG) if (d / TRAIN_LOG).is_file() else None
    stacker_path = d / STACKER
    Verifier(stacker_path, expected_class_names=cand["class_names"])  # sha + class checks
    names = cand["class_names"]
    nal = names.index(NAL) if NAL in names else oof["stack_classes"][0]

    det_by_id = {}
    for info in cand["runs"]:
        for p in load_predictions(info["eval_run"])["photos"]:
            det_by_id[p["capture_id"]] = p
    det_all, casc_all = [], []
    for k in FOLDS:
        casc = cascade_photos(cand, oof, k)
        casc_all += casc
        det_all += [det_by_id[p["capture_id"]] for p in casc]

    primary = paired_recall(det_all, casc_all, nal, ec.bootstrap_n, ec.bootstrap_seed)
    secondary = crossfit_thresholds(cand, oof, det_by_id, nal, ec.fa_reduction)
    lat = latency(cand, stacker_path, exit_run["latency_photos"], ec.latency_warmup,
                  threads=exit_run["threads"])
    lat["gate"] = {
        "verifier_50_p50_ms_max": ec.latency_verifier_p50_ms,
        "total_50_p50_s_max": ec.latency_total_p50_s,
        "passed": bool(lat["verifier_50_ms"]["p50"] <= ec.latency_verifier_p50_ms
                       and lat["total_50_ms"]["p50"] <= ec.latency_total_p50_s * 1000.0),
    }
    ceiling = candidate_ceiling(cand, det_by_id, nal)
    reasons = invalid_reasons(cand, oof, train_log, exit_run)
    decision = decide(reasons, lat["gate"]["passed"], primary["passed"], secondary["passed"])
    earlier = archive_reports(d)
    report = {
        "vrun": vrun, "created": utc_now(), "decision": decision, "invalid_reasons": reasons,
        "rule": "ship iff latency gate AND (primary OR secondary)",
        "preregistration": {"file": PREREG.name,
                            "sha256": sha256_file(PREREG) if PREREG.exists() else None},
        "class": names[nal], "family": cand["family"], "t_star": oof["t_star"],
        "tau_final": oof["final"]["tau"],
        "primary_recall_at_0.5fppi": primary, "secondary_crossfit_thresholds": secondary,
        "latency": lat, "candidate_ceiling": ceiling, "auroc_inner_vs_outer": oof["auroc"],
        "per_fold": {k: {"t_det": oof["folds"][k]["t_det"], "tau": oof["folds"][k]["tau"],
                         "hi": oof["folds"][k]["hi"]} for k in FOLDS},
        "cascade_runs": oof.get("cascade_runs", []),
        "config_sha256": oof["config_sha256"], "candidates_sha256": cand_sha,
        "exit_run": exit_run,
        "git": {step: {"sha": doc.get("git_sha"), "dirty": doc.get("git_dirty")}
                for step, doc in (("mine", cand), ("crossfit", oof), ("exit", exit_run))},
        "earlier_reports": earlier,
    }
    write_json_atomic(d / EXIT_JSON, report)
    write_bytes_atomic(d / EXIT_MD, format_report(report).encode("utf-8"))
    sdoc = read_json(stacker_path)
    sdoc["exit"] = {"decision": decision, "report": EXIT_MD, "reasons": reasons,
                    "decided": report["created"],
                    "earlier_decisions": [e["decision"] for e in earlier]}
    write_json_atomic(stacker_path, sdoc)
    Verifier(stacker_path, expected_class_names=names)
    logger.info(f"exit decision for {vrun}: {decision.upper()}" + (f" ({'; '.join(reasons)})" if reasons else ""))
    return report


def _fmt(v, nd=4):
    return "–" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def format_report(r: dict) -> str:
    p, s, lat, ce, au = (r["primary_recall_at_0.5fppi"], r["secondary_crossfit_thresholds"],
                         r["latency"], r["candidate_ceiling"], r["auroc_inner_vs_outer"])
    ex = r["exit_run"]
    lines = [f"# Verifier exit report — `{r['vrun']}`", ""]
    if r["decision"] == "invalid-smoke":
        lines += ["> **INVALID (smoke)** — no decision; every number below is a plumbing check only:",
                  *[f"> - {x}" for x in r["invalid_reasons"]], ""]
    lines += [
        f"**Decision: {r['decision'].upper()}** — rule: {r['rule']} "
        f"(pre-registered in `{r['preregistration']['file']}`, sha "
        f"{(r['preregistration']['sha256'] or '')[:12]})",
        "",
        f"- family `{r['family']}` · class {r['class']} · anchor t* = {r['t_star']:.5f} · "
        f"tau_final = {r['tau_final']:.5f}",
        f"- cascade eval runs: {', '.join(f'`{x}`' for x in r['cascade_runs']) or '–'}",
        f"- exit run: config `{ex['config']['name']}` (sha {str(ex['config']['sha256'])[:12]}, "
        f"overrides {ex['config']['overrides'] or 'none'}), threads "
        f"{ex['threads'] or 'ORT default'}, {ex['latency_photos']} latency photos · git "
        + ", ".join(f"{k} {str(g['sha'])[:12]}{' (dirty)' if g['dirty'] else ''}"
                    for k, g in r["git"].items()),
    ]
    if r["earlier_reports"]:
        lines.append(
            f"- **{len(r['earlier_reports'])} earlier exit report(s) of this run** (kept in "
            f"`{EXIT_ARCHIVE}/`, every exit is reported): "
            + ", ".join(f"{e['created']} {e['decision']}" for e in r["earlier_reports"]))
    lines += [
        "",
        "## (a) Primary — NAL recall @ 0.5 FPPI, pooled held-out folds",
        "",
    ]
    if p.get("n_gt"):
        lines += [
            f"- detector {p['recall_detector']:.3f} · cascade {p['recall_cascade']:.3f} · "
            f"difference **{p['diff']:+.3f}** (95 % CI {p['diff_ci95']}, paired {p['ci_unit']} "
            f"bootstrap, {p['n_groups']} groups, {p['n_gt']} GT boxes, {p['n_photos']} photos)",
            f"- passed (lower bound > 0): **{p['passed']}**",
        ]
    else:
        lines.append(f"- not evaluable: {p.get('note')} — passed: False")
    pooled = s["pooled"]
    lines += [
        "", "## (b) Secondary — cross-fitted thresholds (held-out folds)", "",
        "| fold | t(k) detector | tau(k) cascade | NAL FA det | NAL FA cascade | added escapes | rescued |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, f in s["per_fold"].items():
        lines.append(f"| {k} | {f['t_det']:.5f} | {f['tau']:.5f} | {f.get('det_fa', 0)} | "
                     f"{f.get('casc_fa', 0)} | {f.get('added_escapes', 0)} | {f.get('rescued', 0)} |")
    lines += [
        "",
        f"- pooled NAL false-alarm boxes: detector {pooled.get('det_fa', 0)}, cascade "
        f"{pooled.get('casc_fa', 0)} (reduction {_fmt(s['nal_fa_reduction'])}, required "
        f">= {s['required_reduction']:.2f})",
        f"- escapes: detector {pooled.get('det_escapes', 0)}, cascade {pooled.get('casc_escapes', 0)} "
        f"of {pooled.get('positives', 0)} positive photos; **added {pooled.get('added_escapes', 0)}**, "
        f"rescued {pooled.get('rescued', 0)}",
        f"- false calls, seated iO: detector {pooled.get('det_false_calls_seated', 0)}, cascade "
        f"{pooled.get('casc_false_calls_seated', 0)} of {pooled.get('negatives_seated', 0)}; handheld: "
        f"detector {pooled.get('det_false_calls_handheld', 0)}, cascade "
        f"{pooled.get('casc_false_calls_handheld', 0)} of {pooled.get('negatives_handheld', 0)}",
        f"- passed: **{s['passed']}**",
        "", "## (c) Latency gate (ONNX Runtime CPU)", "",
        "| | p50 ms | p95 ms |", "|---|---|---|",
        f"| detector (20 tiles) | {lat['detector_ms']['p50']} | {lat['detector_ms']['p95']} |",
        f"| verifier, this photo's candidates | {lat['verifier_actual_ms']['p50']} | {lat['verifier_actual_ms']['p95']} |",
        f"| verifier, {lat['n_candidates']} candidates | {lat['verifier_50_ms']['p50']} | {lat['verifier_50_ms']['p95']} |",
        f"| detector + verifier({lat['n_candidates']}) | {lat['total_50_ms']['p50']} | {lat['total_50_ms']['p95']} |",
        f"| reference: verifier, {lat['n_candidates']} NAL-sized boxes (s = 1) | "
        f"{lat['verifier_50_native_ms']['p50']} | {lat['verifier_50_native_ms']['p95']} |",
        "",
        f"- gate: verifier p50 <= {lat['gate']['verifier_50_p50_ms_max']} ms and total p50 <= "
        f"{lat['gate']['total_50_p50_s_max']} s on {lat['n_photos']} photos -> **{lat['gate']['passed']}**",
        f"- {lat['note']}; host {lat.get('host')} — timings are only meaningful on an otherwise idle machine",
        "", "## (d) Candidate ceiling", "",
        f"- NAL GT boxes matched (IoU 0.3) by a top-{ce['top_k']} candidate: {_fmt(ce['topk_share'])}; "
        f"by any detection at the floor: {_fmt(ce['all_detections_share'])} ({ce['n_gt']} boxes)",
        "", "## (e) Inner vs outer out-of-fold AUROC (NAL)", "",
        f"- verifier: inner {_fmt(au.get('verifier_inner'))} · outer {_fmt(au.get('verifier_outer'))}; "
        f"stacker: inner {_fmt(au.get('stacker_inner'))} · outer {_fmt(au.get('stacker_outer'))}",
        "",
        "_(d) and (e) are descriptive. Photo-level numbers only; crop-level verifier "
        "accuracy is never compared with detector metrics._",
    ]
    return "\n".join(lines) + "\n"
