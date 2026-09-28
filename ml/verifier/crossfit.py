"""
Nested cross-fit: out-of-fold verifier AND stacker scores for every fold, plus
the final artifacts.

    python verifier/cli.py crossfit --run-name v1 --device mps --operating-point <op.json>

Outer fold k with inner folds {i, j} (folds are the detector's own folds):
  V(k,-i) trains on j and scores i;  V(k,-j) trains on i and scores j
      -> inner out-of-fold verifier scores on i and j
  S_k   = logistic stacker fitted on those inner OOF scores (+ det features)
  t(k)  = the choose-op rule (photo sensitivity >= 0.95) on the detector's
          predictions of i and j — the cross-fitted detector threshold
  tau(k)= select_tau on i and j at t(k)
  V_k   trains on i u j and scores k  ->  cascade score on k = S_k(V_k)
No model, stacker or threshold used on fold k ever saw fold k. A verifier
trained on ONE fold is the same model whichever outer fold uses it, so the six
inner models are the three single-fold models U_a (trained once each).

Why nested: a stacker fitted on the SAME verifier scores it is later applied
to learns how much to trust an in-sample verifier (overconfident) — its
weights would be optimistic and the cascade's out-of-fold number biased.
Fitting S_k on inner out-of-fold scores keeps every number honest; the price
is that S_k sees scores of verifiers trained on 1/3 of the data and is
applied to one trained on 2/3 (reported as inner-vs-outer AUROC, exit (e)).

Final artifacts: V_final on all three folds (-> verifier_final.pt, exported to
verifier.onnx by export_onnx.py); S_final on the pooled OUTER out-of-fold
scores; tau_final by select_tau on those pooled scores at the family's frozen
t* (operating_point.json), which is also the anchor of every final score.

Cascade eval runs <F>+<vrun>_fold<k> are written through run_eval._write_outputs
with scores = A(S_k(V_k); tau(k), t*) (hi-passthrough applied), det_scores
kept and predictions.json v2 source.kind = cascade.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from config.settings import settings
from dataprep.split import load_split
from eval.photo_scorer import threshold_for_sensitivity
from infer.verifier import candidate_features, final_scores, stack_proba
from verifier.cascade_eval import write_run
from verifier.common import (
    FINAL_WEIGHTS, FOLDS, OOF_SCORES, TRAIN_LOG, Refusal, git_state, host, read_json,
    sha256_file, utc_now, vrun_dir, write_json_atomic,
)
from verifier.config import VerifierConfig, load_config
from verifier.crops import load_windows
from verifier.mine import load_candidates
from verifier.stacker import auroc, fit_class_stacker, resolve_hi, select_tau

logger = logging.getLogger("ml.verifier.crossfit")

NAL = "Nicht ausreichend Lot"


# ------------------------------------------------------------------ the plan


def make_plan(folds=FOLDS) -> dict:
    folds = list(folds)
    if len(folds) != 3:
        raise ValueError("the nested cross-fit needs exactly three folds")
    models = {f"U_{a}": {"train": [a]} for a in folds}
    models.update({f"V_{k}": {"train": [f for f in folds if f != k]} for k in folds})
    models["V_final"] = {"train": list(folds)}
    outer = {}
    for k in folds:
        i, j = [f for f in folds if f != k]
        outer[k] = {
            "inner": [i, j],
            # inner fold -> the single-fold model that scores it (trained on the other inner fold)
            "inner_scorer": {i: f"U_{j}", j: f"U_{i}"},
            "stacker_rows": [i, j],
            "threshold_folds": [i, j],
            "outer_model": f"V_{k}",
        }
    return {"folds": folds, "models": models, "outer": outer}


def check_plan(plan: dict) -> None:
    """The nesting invariants (tests/test_verifier_mining.py exercises them)."""
    models = plan["models"]
    for k, o in plan["outer"].items():
        used = [o["outer_model"]] + list(o["inner_scorer"].values())
        for m in used:
            if k in models[m]["train"]:
                raise AssertionError(f"model {m} used on outer fold {k} was trained on it")
        if k in o["stacker_rows"] or k in o["threshold_folds"]:
            raise AssertionError(f"outer fold {k}'s stacker/threshold uses fold {k}")
        for f, m in o["inner_scorer"].items():
            if f in models[m]["train"]:
                raise AssertionError(f"inner fold {f} is scored by {m}, which trained on it")
            if f not in o["stacker_rows"]:
                raise AssertionError(f"inner scorer for {f} outside the stacker rows")


def stacker_rows(plan: dict, k: str, row_ids, row_folds) -> list[tuple[int, str]]:
    """(row id, single-fold model that scores it) for the rows S_k is fitted on:
    only rows of k's inner folds, each scored by the model trained on the OTHER
    inner fold (so no stacker row carries an in-sample verifier score)."""
    o = plan["outer"][k]
    return [(int(r), o["inner_scorer"][f]) for r, f in zip(row_ids, row_folds)
            if f in o["stacker_rows"]]


# ------------------------------------------------------------------ helpers


def load_operating_point(path: Optional[Path], cand: dict) -> tuple[float, dict]:
    """t* = the family's frozen detector threshold (operating_point.json)."""
    schema, family = cand["schema"], cand["family"]
    if path is None:
        path = settings.eval_dir / f"operating_point_{schema}_{family}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"no operating point for family {family!r} ({path}); run `cli.py choose-op` "
                "on the three fold runs or pass --operating-point"
            )
    path = Path(path)
    op = read_json(path)
    if op.get("schema") != schema:
        raise ValueError(f"operating point is for schema {op.get('schema')!r}, not {schema!r}")
    if op.get("class_names") and list(op["class_names"]) != list(cand["class_names"]):
        raise ValueError("operating point class_names differ from the candidates'")
    runs = sorted(r["eval_run"] for r in cand["runs"])
    same = (op.get("family") == family) if op.get("family") else sorted(op.get("runs") or []) == runs
    if not same:
        msg = (f"operating point {path.name} belongs to family {op.get('family')!r} / runs "
               f"{op.get('runs')}, not {family!r} / {runs}")
        if not cand["smoke"]:
            raise Refusal(msg)
        logger.warning("SMOKE: " + msg)
    t = float(op["threshold"])
    if not (0.0 < t < 1.0):
        raise ValueError(f"operating point threshold {t} not in (0, 1)")
    return t, {"file": str(path.resolve()), "sha256": sha256_file(path), "threshold": t,
               "family": op.get("family"), "runs": op.get("runs"), "rule": op.get("rule")}


def _photo_index(cand: dict) -> dict:
    return {p["capture_id"]: p for p in cand["photos"]}


def _features(rows: np.ndarray, ver: np.ndarray, cands: list[dict], photos: dict) -> np.ndarray:
    x = np.zeros((len(rows), 4))
    by_photo = defaultdict(list)
    for n, r in enumerate(rows):
        by_photo[cands[r]["capture_id"]].append(n)
    for cid, idx in by_photo.items():
        ph = photos[cid]
        rr = [rows[n] for n in idx]
        x[idx] = candidate_features([cands[r]["det_score"] for r in rr], ver[idx],
                                    [cands[r]["box"] for r in rr], ph["width"], ph["height"])
    return x


def _tau_photos(photo_ids, cands, photos, p_of: dict, hi: dict, stack: list[int]) -> list[dict]:
    """select_tau input for the given photos: p_of[row] for stacked candidates."""
    per = {cid: {"capture_id": cid, "positive": bool(photos[cid]["gt"]),
                 "det_photo_score": photos[cid]["det_photo_score"],
                 "stacked": [], "passthrough": []} for cid in photo_ids}
    for c in cands:
        if c["kind"] == "gt" or c["capture_id"] not in per:
            continue
        rec = per[c["capture_id"]]
        if c["class"] in stack:
            rec["stacked"].append((p_of[c["id"]], c["det_score"], hi.get(c["class"])))
        else:
            rec["passthrough"].append(c["det_score"])
    return list(per.values())


def detector_threshold(photo_ids, photos, target: float) -> Optional[float]:
    """The choose-op rule on the detector's own photo scores."""
    recs = [{"gt": photos[cid]["gt"], "scores": [photos[cid]["det_photo_score"]]}
            for cid in photo_ids]
    return threshold_for_sensitivity(recs, target)


# ------------------------------------------------------------------ run


def run(
    vrun: str,
    device: str = "auto",
    cfg: Optional[VerifierConfig] = None,
    operating_point: Optional[Path] = None,
    write_eval_runs: bool = True,
) -> dict:
    import torch

    from verifier.train import pick_device, predict, train_model

    cand, cand_sha = load_candidates(vrun)
    cfg = cfg or load_config()
    mined = cand["config"]["candidates"]
    for key in ("top_k", "iou_positive", "min_pos_per_class"):
        if mined[key] != getattr(cfg.candidates, key):
            raise ValueError(f"config candidates.{key}={getattr(cfg.candidates, key)} differs "
                             f"from the mined candidates ({mined[key]}) — rerun mine")
    windows = load_windows(vrun, cand_sha)
    cands, photos = cand["candidates"], _photo_index(cand)
    class_names = cand["class_names"]
    n_cls = len(class_names)
    stack = list(cand["stack_classes"])
    if not stack:
        raise RuntimeError("no stacked class (every class is passthrough) — nothing to cross-fit")
    t_star, op_info = load_operating_point(operating_point, cand)
    dev = pick_device(device)
    plan = make_plan()
    check_plan(plan)
    d = vrun_dir(vrun)
    t_run = time.perf_counter()

    kind = np.array([c["kind"] for c in cands])
    fold = np.array([c["fold"] for c in cands])
    cls = np.array([c["class"] for c in cands], dtype=np.int64)
    label = np.array([c["label"] for c in cands], dtype=np.int64)
    det = np.array([np.nan if c["det_score"] is None else c["det_score"] for c in cands])
    is_stack = np.isin(cls, stack)
    train_mask = is_stack & (label != -1)
    score_mask = is_stack & (kind != "gt")

    logs, ver_inner, ver_outer = [], {}, {}
    seeds = {name: cfg.train.seed + n for n, name in enumerate(sorted(plan["models"]))}

    def train(name: str):
        folds_ = plan["models"][name]["train"]
        ids = np.nonzero(train_mask & np.isin(fold, folds_))[0]
        model, log = train_model(name, windows, ids, label[ids], cls[ids], n_cls, cfg, dev,
                                 seed=seeds[name])
        log["train_folds"] = folds_
        logs.append(log)
        return model

    # single-fold models: the inner out-of-fold scores
    u_scores: dict[str, dict[int, float]] = {}
    for a in FOLDS:
        name = f"U_{a}"
        model = train(name)
        rows = np.nonzero(score_mask & ~np.isin(fold, plan["models"][name]["train"]))[0]
        u_scores[name] = dict(zip(rows.tolist(), predict(model, windows, rows, cls[rows], dev).tolist()))
        del model
    # two-fold models: the outer out-of-fold scores
    for k in FOLDS:
        model = train(f"V_{k}")
        rows = np.nonzero(score_mask & (fold == k))[0]
        ver_outer.update(zip(rows.tolist(), predict(model, windows, rows, cls[rows], dev).tolist()))
        del model
    model = train("V_final")
    final_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": final_state, "num_classes": n_cls,
                "embed_dim": cfg.model.embed_dim, "dropout": cfg.model.dropout,
                "weights_source": model.weights_source}, d / FINAL_WEIGHTS)
    weights_source = model.weights_source
    del model

    rows_all = np.nonzero(score_mask)[0]
    ver_out_arr = np.array([ver_outer[r] for r in rows_all])
    x_outer = _features(rows_all, ver_out_arr, cands, photos)
    pos_of = {r: n for n, r in enumerate(rows_all)}

    folds_doc, p_outer, final_outer, p_inner_all = {}, {}, {}, defaultdict(dict)
    split = load_split()
    for k in FOLDS:
        o = plan["outer"][k]
        pairs = stacker_rows(plan, k, rows_all.tolist(), [str(fold[r]) for r in rows_all])
        inner_rows = np.array([r for r, _ in pairs], dtype=np.int64)
        ver_in = np.array([u_scores[m][r] for r, m in pairs])
        for r, z in zip(inner_rows.tolist(), ver_in.tolist()):
            ver_inner.setdefault(r, {})[k] = z
        x_in = _features(inner_rows, ver_in, cands, photos)
        inner_photos = [cid for cid, p in photos.items() if p["fold"] in o["inner"]]
        t_k = detector_threshold(inner_photos, photos, cfg.exit.target_sensitivity)
        if t_k is None:
            raise RuntimeError(f"outer {k}: no positive photo with a detection on {o['inner']}")
        params, fits, his, p_in = {}, {}, {}, np.zeros(len(inner_rows))
        for c in stack:
            m = cls[inner_rows] == c
            fit_m = m & (label[inner_rows] != -1)
            params[c], fits[c] = fit_class_stacker(x_in[fit_m], label[inner_rows][fit_m], cfg.stacker.C)
            his[c] = resolve_hi(cfg.stacker.hi_passthrough, det[inner_rows][fit_m],
                                label[inner_rows][fit_m], t_k, cfg.stacker.hi_min_pos)
            p_in[m] = stack_proba(params[c], x_in[m])
        p_of = dict(zip(inner_rows.tolist(), p_in.tolist()))
        for r, p in p_of.items():
            p_inner_all[r][k] = p
        tau_k, tau_info = select_tau(_tau_photos(inner_photos, cands, photos, p_of, his, stack), t_k)
        # outer fold: V_k scores through S_k, anchored at tau(k) -> global t*
        out_rows = [r for r in rows_all if fold[r] == k]
        for c in stack:
            rr = [r for r in out_rows if cls[r] == c]
            if not rr:
                continue
            xo = x_outer[[pos_of[r] for r in rr]]
            po = stack_proba(params[c], xo)
            fo = final_scores({**params[c], "tau": tau_k, "hi_passthrough": his[c]}, po, det[rr], t_star)
            p_outer.update(zip(rr, po.tolist()))
            final_outer.update(zip(rr, fo.tolist()))
        sdoc = {
            "kind": "zander-stacker-fold", "vrun": vrun, "outer_fold": k, "inner_folds": o["inner"],
            "t_star": t_star, "t_det": t_k, "tau": tau_k, "tau_info": tau_info,
            "features": ["det_logit", "ver_logit", "log_area", "seam_margin"],
            "classes": {class_names[c]: {"mode": "stack", **params[c], "tau": tau_k,
                                         "hi_passthrough": his[c], "fit": fits[c]} for c in stack}
                       | {class_names[c]: {"mode": "passthrough"} for c in range(n_cls) if c not in stack},
            "verifier": f"V_{k} (torch, trained on {o['inner']})", "created": utc_now(),
        }
        sfile = d / f"stacker_{k}.json"
        ssha = write_json_atomic(sfile, sdoc)
        folds_doc[k] = {"inner": o["inner"], "t_det": t_k, "tau": tau_k, "tau_info": tau_info,
                        "hi": {class_names[c]: his[c] for c in stack},
                        "stacker": {class_names[c]: params[c] for c in stack},
                        "fit": {class_names[c]: fits[c] for c in stack},
                        "stacker_file": sfile.name, "stacker_sha256": ssha}
        logger.info(f"outer {k}: t_det={t_k:.5f} tau={tau_k:.5f} hi={his} "
                    f"({tau_info['constraints']} constraining photos)")

    # final stacker on the pooled OUTER out-of-fold scores
    final_params, final_fit, final_hi, p_fin = {}, {}, {}, np.zeros(len(rows_all))
    for c in stack:
        m = cls[rows_all] == c
        fit_m = m & (label[rows_all] != -1)
        final_params[c], final_fit[c] = fit_class_stacker(x_outer[fit_m], label[rows_all][fit_m],
                                                          cfg.stacker.C)
        final_hi[c] = resolve_hi(cfg.stacker.hi_passthrough, det[rows_all][fit_m],
                                 label[rows_all][fit_m], t_star, cfg.stacker.hi_min_pos)
        p_fin[m] = stack_proba(final_params[c], x_outer[m])
    p_final = dict(zip(rows_all.tolist(), p_fin.tolist()))
    tau_final, tau_final_info = select_tau(
        _tau_photos(list(photos), cands, photos, p_final, final_hi, stack), t_star)

    nal = class_names.index(NAL) if NAL in class_names else stack[0]
    au = _aurocs(rows_all, label, cls, fold, ver_inner, ver_outer, p_inner_all, p_outer, nal)
    oof = {
        "format_version": 1, "kind": "zander-verifier-oof", "vrun": vrun, "created": utc_now(),
        "candidates_sha256": cand_sha, "smoke": cand["smoke"], "fraction": cand["fraction"],
        "class_names": class_names, "stack_classes": stack, "t_star": t_star,
        "operating_point": op_info, "plan": plan, "folds": folds_doc,
        "final": {"t_star": t_star, "tau": tau_final, "tau_info": tau_final_info,
                  "hi": {class_names[c]: final_hi[c] for c in stack},
                  "stacker": {class_names[c]: final_params[c] for c in stack},
                  "fit": {class_names[c]: final_fit[c] for c in stack}},
        "auroc": au, "weights_source": weights_source, "device": str(dev),
        "config": cfg.to_dict(), "config_sha256": cfg.sha256, **git_state(), **host(),
        "rows": [
            {"id": int(r), "fold": str(fold[r]), "class": int(cls[r]), "label": int(label[r]),
             "det": float(det[r]), "ver_inner": ver_inner.get(int(r), {}),
             "ver_outer": ver_outer[int(r)], "p_inner": p_inner_all.get(int(r), {}),
             "p_outer": p_outer[int(r)], "final_outer": final_outer[int(r)],
             "p_final": p_final[int(r)]}
            for r in rows_all
        ],
        "seconds": round(time.perf_counter() - t_run, 1),
    }
    write_json_atomic(d / OOF_SCORES, oof, indent=None)
    write_json_atomic(d / TRAIN_LOG, {"vrun": vrun, "device": str(dev), "created": utc_now(),
                                      "models": logs, "seconds": oof["seconds"]})
    logger.info(f"final: tau={tau_final:.5f} at t*={t_star:.5f}, hi={final_hi}; AUROC {au}")
    if write_eval_runs:
        oof["cascade_runs"] = write_cascade_runs(vrun, cand, oof, t_star, split)
        write_json_atomic(d / OOF_SCORES, oof, indent=None)
    return oof


def _aurocs(rows_all, label, cls, fold, ver_inner, ver_outer, p_inner, p_outer, c) -> dict:
    """(e): verifier (and stacker) AUROC on inner vs outer out-of-fold scores."""
    rows = [int(r) for r in rows_all if cls[r] == c and label[r] in (0, 1)]
    out = {"class": int(c)}
    inner_z, inner_y, inner_p = [], [], []
    for r in rows:
        for k, z in ver_inner.get(r, {}).items():
            inner_z.append(z)
            inner_y.append(label[r])
            inner_p.append(p_inner[r][k])
    out["verifier_inner"] = auroc(inner_z, inner_y)
    out["verifier_outer"] = auroc([ver_outer[r] for r in rows], [label[r] for r in rows])
    out["stacker_inner"] = auroc(inner_p, inner_y)
    out["stacker_outer"] = auroc([p_outer[r] for r in rows], [label[r] for r in rows])
    out["per_fold_verifier_outer"] = {
        k: auroc([ver_outer[r] for r in rows if fold[r] == k], [label[r] for r in rows if fold[r] == k])
        for k in FOLDS
    }
    return out


def cascade_photos(cand: dict, oof: dict, k: str) -> list[dict]:
    """Photo records of outer fold k (detector-run order): top-k candidates with
    final scores (stacked: A(S_k(V_k)); passthrough: detector score)."""
    finals = {r["id"]: r["final_outer"] for r in oof["rows"]}
    by_photo = defaultdict(list)
    for c in cand["candidates"]:
        if c["kind"] != "gt" and c["fold"] == k:
            by_photo[c["capture_id"]].append(c)
    out = []
    for ph in cand["photos"]:
        if ph["fold"] != k:
            continue
        cs = by_photo.get(ph["capture_id"], [])
        sc = [finals.get(c["id"], c["det_score"]) for c in cs]
        order = sorted(range(len(cs)), key=lambda i: (-sc[i], i))
        out.append({
            "capture_id": ph["capture_id"], "group": ph["group"], "board": ph["board"],
            "pose": ph["pose"], "gt_verdict": ph["gt_verdict"],
            "gt": [(c, tuple(b)) for c, b in ph["gt"]],
            "other_gt": [(n, tuple(b)) for n, b in ph["other_gt"]],
            "boxes": [tuple(cs[i]["box"]) for i in order],
            "scores": [float(sc[i]) for i in order],
            "labels": [int(cs[i]["class"]) for i in order],
            "det_scores": [float(cs[i]["det_score"]) for i in order],
            "width": ph["width"], "height": ph["height"],
        })
    return out


def write_cascade_runs(vrun: str, cand: dict, oof: dict, t_star: float, split: dict) -> list[str]:
    names = []
    probe_all = split.get("probe_boxes", [])
    for info in cand["runs"]:
        k = info["bucket"]
        run_name = f"{cand['family']}+{vrun}_{k}"
        photos = cascade_photos(cand, oof, k)
        ids = {p["capture_id"] for p in photos}
        probe = [p for p in probe_all if p["capture_id"] in ids and p["class"] in cand["class_names"]]
        source = {
            "kind": "cascade", "bundle_run": info.get("bundle_run"), "family": cand["family"],
            "fold": info.get("fold"), "model_sha256": info.get("model_sha256"),
            "model_resolved": info.get("model_resolved"), "adapter": info.get("adapter"),
            "conf_floor": info.get("conf_floor"), "verifier_run": vrun,
            "stacker_sha256": oof["folds"][k]["stacker_sha256"],
            "detector_eval_run": info["eval_run"],
        }
        base = {
            "model": f"{info.get('model')}+{vrun}", "adapter": info.get("adapter"),
            "schema": cand["schema"],
            "mean_ms_per_photo": None,
            "latency_note": "offline cross-fit scoring (torch V_k); see exit_report latency",
            "tiler_version": info.get("tiler_version"), "split_id": info.get("split_id"),
            "snapshot_id": info.get("snapshot_id"),
            "dataset_fingerprint": info.get("dataset_fingerprint"), "host": host()["host"],
            "fraction": cand["fraction"],
            "cascade": {"tau": oof["folds"][k]["tau"], "t_det_crossfit": oof["folds"][k]["t_det"],
                        "t_star": t_star, "hi": oof["folds"][k]["hi"]},
        }
        write_run(run_name, photos, cand["class_names"], k, t_star,
                  f"frozen anchor t* ({Path(oof['operating_point']['file']).name}); "
                  f"verifier tau({k}) chosen on {oof['folds'][k]['inner']}",
                  base, source, smoke=cand["smoke"], probe=probe,
                  record=False, gallery=False)  # OOF analysis runs: no results row, no gallery
        names.append(run_name)
    return names
