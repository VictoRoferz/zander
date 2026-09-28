"""
Paired comparison of two eval runs — or of two FAMILIES pooled over their
out-of-fold runs — on the SAME photos: the report's G1 selection rule and the
def3-vs-nal1 non-inferiority gate.

    python cli.py compare yolo26n_def3 yolo11s_def3 --class NAL             # families (pooled OOF)
    python cli.py compare yolo26n_def3 yolo26n_nal1 --class NAL --margin 0.05
    python cli.py compare <run_a> <run_b> --class any                       # two eval runs

Operands: an eval run name (ml_data/eval/<name>/), else a family `F`, which
resolves to `F_fold0..2` (CONTRACTS §8.2) pooled by concatenation in fold
order; both sides must then cover identical capture_ids in identical order.
Pooling refuses fold runs that are provably not out-of-fold (the same model
on several folds, or a model scored off its own fold).

`--class` takes a class name, its short name (NAL / Brücke / Fahne), an index
(meaning run A's class; B must have the same class) or `any` (class-agnostic,
same schema only) — resolved per side, so NAL works for def3 vs nal1.

G1: a candidate only beats the runner-up if the paired difference's 95 % CI
excludes 0; otherwise the faster/simpler model wins. With `--margin δ` the
non-inferiority verdict is added: A is non-inferior to B iff the CI's lower
bound of (A − B) > −δ.

Paired = both models are scored on identical photos, so photo-to-photo
difficulty cancels out. Uncertainty comes from a cluster bootstrap over the
split_v2 board GROUPS (re-shots of one board are not independent) and, for the
photo verdicts, an exact McNemar test on the discordant photos. With few
discordant photos McNemar cannot reach significance (<= 5 discordant: never at
alpha 0.05) — which is the honest answer on this dataset size.

Escape / verdict numbers use each side's FROZEN operating point (the one a
test/probe run was evaluated with, else operating_point_<schema>_<family>.json,
else a legacy op file listing the run), "n/a" when there is none — never a
self-tuned threshold. They are computed on the photos positive under BOTH
sides' schemas (def3 vs nal1: the NAL photos), so the escape is truly paired.

Writes ml_data/eval/compare_<A>__<B>__<class>.json.
"""
from __future__ import annotations

import json
import logging
import math
import random
import re
from pathlib import Path

from config.settings import settings
from dataprep.remap import SHORT
from dataprep.split import load_split
from eval.photo_scorer import class_records, is_negative, is_positive, pscore
from eval.run_eval import EvalGuardError, load_predictions, run_source
from eval.scorer import (
    BOOTSTRAP_N,
    FPPI_OPERATING,
    IOU_OPERATING,
    _froc_from_stats,
    _photo_stats,
    cluster_resample,
    recall_at_fppi,
)

logger = logging.getLogger("ml.compare")

CLASS_ANY = "any"
NA = "n/a"


class CompareError(ValueError):
    """The two sides cannot be compared (class, photos, order)."""


def _threshold(run: str) -> float | None:
    m = json.loads((settings.eval_dir / run / "metrics.json").read_text(encoding="utf-8"))
    return m["threshold"]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def class_slug(name: str) -> str:
    """File-name-safe class label: NAL, Bruecke, Fahne, any."""
    s = SHORT.get(name, name)
    s = s.replace("ü", "ue").replace("ä", "ae").replace("ö", "oe").replace("ß", "ss")
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s)


def resolve_class(spec, class_names: list[str]) -> tuple[int | None, str]:
    """A --class spec -> (index in `class_names`, canonical name); `any` -> (None, "any")."""
    if spec is None:
        spec = 0
    if isinstance(spec, int) or (isinstance(spec, str) and spec.strip().isdigit()):
        i = int(spec)
        if not 0 <= i < len(class_names):
            raise CompareError(f"class index {i} out of range for {class_names}")
        return i, class_names[i]
    s = str(spec).strip().casefold()
    if s in (CLASS_ANY, "all", "agnostic"):
        return None, CLASS_ANY
    for i, name in enumerate(class_names):
        if s in {name.casefold(), SHORT.get(name, name).casefold(), class_slug(name).casefold()}:
            return i, name
    raise CompareError(f"class {spec!r} is not in {class_names}")


def _family_runs(name: str) -> list[str]:
    k = len(load_split()["folds"])
    return [f"{name}_fold{i}" for i in range(k)]


def resolve(name: str) -> dict:
    """An eval run, or a family pooled over its fold runs (concatenated in fold order)."""
    if (settings.eval_dir / name / "predictions.json").is_file():
        doc = load_predictions(name)
        return {"name": name, "kind": "run", "runs": [name], "bucket": doc["bucket"],
                "schema": doc["schema"], "class_names": doc["class_names"],
                "photos": doc["photos"], "docs": [doc]}
    runs = _family_runs(name)
    missing = [r for r in runs if not (settings.eval_dir / r / "predictions.json").is_file()]
    if missing:
        raise CompareError(
            f"{name!r} is neither an eval run nor a complete family (missing {missing} "
            f"in {settings.eval_dir})"
        )
    docs = [load_predictions(r) for r in runs]
    problems = []
    for i, (r, d) in enumerate(zip(runs, docs)):
        if d["bucket"] != f"fold{i}":
            problems.append(f"{r} scored bucket {d['bucket']!r}, not fold{i}")
        if (d["schema"], d["class_names"]) != (docs[0]["schema"], docs[0]["class_names"]):
            problems.append(f"{r} has schema {d['schema']!r}, the family {docs[0]['schema']!r}")
        if d.get("guard_overridden"):
            problems.append(f"{r} was evaluated with guards overridden")
    sources = [run_source(d) for d in docs]
    shas = [s.get("model_sha256") for s in sources]
    if all(shas) and len(set(shas)) != len(shas):
        problems.append("the same model was scored on several folds — not out-of-fold")
    for r, d, s in zip(runs, docs, sources):
        if s.get("fold") and s["fold"] != d["bucket"]:
            problems.append(f"{r}: its model validates on {s['fold']}, not {d['bucket']}")
    if problems:
        raise EvalGuardError(f"family {name!r} cannot be pooled: " + "; ".join(problems))
    if not all(shas) or not all(s.get("fold") for s in sources):
        logger.warning(f"family {name}: provenance of some fold runs unverifiable (v1 predictions)")
    photos = [p for d in docs for p in d["photos"]]
    ids = [p["capture_id"] for p in photos]
    if len(set(ids)) != len(ids):
        raise EvalGuardError(f"family {name!r}: fold runs overlap (a capture appears twice)")
    return {"name": name, "kind": "family", "runs": runs,
            "bucket": "oof(" + "+".join(d["bucket"] for d in docs) + ")",
            "schema": docs[0]["schema"], "class_names": docs[0]["class_names"],
            "photos": photos, "docs": docs}


def _family_of(res: dict) -> str | None:
    if res["kind"] == "family":
        return res["name"]
    src = run_source(res["docs"][0])
    if src.get("family"):
        return src["family"]
    from registry import family_from_run_name

    return family_from_run_name(res["name"].split("__")[0])


def frozen_operating_point(res: dict) -> tuple[float | None, str]:
    """(threshold, source file) of a resolved side's frozen operating point, or (None, "n/a")."""
    if res["kind"] == "run":
        mpath = settings.eval_dir / res["name"] / "metrics.json"
        if mpath.is_file():
            m = json.loads(mpath.read_text(encoding="utf-8"))
            if m.get("operating_point"):
                return m["operating_point"]["threshold"], Path(m["operating_point"]["file"]).name
            if str(m.get("threshold_source", "")).startswith("frozen:"):
                return m["threshold"], m["threshold_source"][len("frozen:"):].strip()
    family = _family_of(res)
    path = settings.eval_dir / f"operating_point_{res['schema']}_{family}.json"
    if family and path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))["threshold"], path.name
    # legacy / renamed runs: an op file that lists these runs, or (single run)
    # whose fold models include this run's model file
    sha = run_source(res["docs"][0]).get("model_sha256") if res["kind"] == "run" else None
    for path in sorted(settings.eval_dir.glob(f"operating_point_{res['schema']}_*.json")):
        op = json.loads(path.read_text(encoding="utf-8"))
        if set(res["runs"]) <= set(op.get("runs") or []):
            return op["threshold"], path.name
        if sha and sha in _op_model_shas(op):
            return op["threshold"], path.name
    return None, NA


def _op_model_shas(op: dict) -> set[str]:
    from registry import file_sha256, resolve_model_path

    shas = {s.get("model_sha256") for s in op.get("sources") or []}
    for m in op.get("models") or []:
        path = resolve_model_path(m)
        if path is not None:
            shas.add(file_sha256(path))
    return shas - {None}


def _ci(xs: list[float]) -> list[float] | None:
    if not xs:
        return None
    xs = sorted(xs)
    return [round(xs[int(0.025 * len(xs))], 4), round(xs[int(0.975 * len(xs)) - 1], 4)]


def compare(
    a: str, b: str, cls="NAL", margin: float | None = None, seed: int = 0,
    *, thresholds: tuple[float | None, float | None] | None = None, write: bool = True,
    resolved: tuple[dict, dict] | None = None,
) -> dict:
    """Paired comparison of runs/families `a` and `b` (see module doc).

    `thresholds` overrides the auto-discovered frozen operating points (tests);
    `resolved` passes already resolved sides (summary)."""
    A, B = resolved or (resolve(a), resolve(b))
    ids_a = [p["capture_id"] for p in A["photos"]]
    if ids_a != [p["capture_id"] for p in B["photos"]] or A["bucket"] != B["bucket"]:
        raise CompareError(
            "runs must cover the same bucket and photos in the same order "
            f"({A['name']}: {A['bucket']}, {len(ids_a)} photos; "
            f"{B['name']}: {B['bucket']}, {len(B['photos'])} photos)"
        )
    ia, cname = resolve_class(cls, A["class_names"])
    if ia is None:
        if A["class_names"] != B["class_names"]:
            raise CompareError("--class any needs identical class schemas; name a class (e.g. NAL)")
        ib = None
    else:
        ib, _ = resolve_class(cname, B["class_names"])
    photos_a, photos_b = A["photos"], B["photos"]
    clusters = [p["group"] for p in photos_a]
    if thresholds is not None:
        (ta, tb), (op_a, op_b) = thresholds, ("explicit", "explicit")
    else:
        (ta, op_a), (tb, op_b) = frozen_operating_point(A), frozen_operating_point(B)
    stats_a = _photo_stats(class_records(photos_a, ia), IOU_OPERATING)
    stats_b = _photo_stats(class_records(photos_b, ib), IOU_OPERATING)
    # photos positive under BOTH schemas (same schema: plain positives)
    joint = [is_positive(pa) and is_positive(pb) for pa, pb in zip(photos_a, photos_b)]

    def recall(stats, idx):
        pts, n_gt = _froc_from_stats([stats[i] for i in idx])
        return recall_at_fppi(pts, FPPI_OPERATING)[0] if n_gt else None

    def escape(photos, t, idx):
        pos = [photos[i] for i in idx if joint[i]]
        return sum(1 for p in pos if pscore(p) < t) / len(pos) if pos and t is not None else None

    full = list(range(len(photos_a)))
    ra, rb = recall(stats_a, full), recall(stats_b, full)
    if ra is None or rb is None:
        raise CompareError(f"class {cname!r} has no ground-truth boxes on these photos")
    ea, eb = escape(photos_a, ta, full), escape(photos_b, tb, full)
    point = {
        "recall_diff": ra - rb,
        "escape_diff": ea - eb if ea is not None and eb is not None else None,
    }
    rng = random.Random(seed)
    boots = {"recall_diff": [], "escape_diff": []}
    for _ in range(BOOTSTRAP_N):
        idx = cluster_resample(len(photos_a), clusters, rng)
        r1, r2 = recall(stats_a, idx), recall(stats_b, idx)
        if r1 is not None and r2 is not None:
            boots["recall_diff"].append(r1 - r2)
        e1, e2 = escape(photos_a, ta, idx), escape(photos_b, tb, idx)
        if e1 is not None and e2 is not None:
            boots["escape_diff"].append(e1 - e2)

    # McNemar on photo verdict correctness (joint positives + iO negatives)
    nb = nc = 0
    if ta is not None and tb is not None:
        for pa, pb, pos in zip(photos_a, photos_b, joint):
            if not (pos or is_negative(pa)):
                continue
            ok_a = (pscore(pa) >= ta) == pos
            ok_b = (pscore(pb) >= tb) == pos
            nb += ok_a and not ok_b
            nc += ok_b and not ok_a
    result = {
        "run_a": A["name"], "run_b": B["name"], "bucket": A["bucket"],
        "class": cname,
        "recall_at_0.5fppi_diff": round(point["recall_diff"], 4),
        "recall_diff_ci95": _ci(boots["recall_diff"]),
        "escape_diff": None if point["escape_diff"] is None else round(point["escape_diff"], 4),
        "escape_diff_ci95": _ci(boots["escape_diff"]),
        "verdict_discordant": {"a_right_b_wrong": nb, "b_right_a_wrong": nc},
        "mcnemar_p": round(mcnemar_exact(nb, nc), 4) if ta is not None and tb is not None else None,
        "ci_unit": "board group",
    }
    rc = result["recall_diff_ci95"]
    decisive = rc is not None and (rc[0] > 0 or rc[1] < 0)
    result["g1_decision"] = (
        ("A better" if rc[0] > 0 else "B better") if decisive
        else "no significant difference -> prefer the faster/simpler model"
    )
    result.update({
        "kind": f"{A['kind']} vs {B['kind']}",
        "runs_a": A["runs"], "runs_b": B["runs"],
        "schema_a": A["schema"], "schema_b": B["schema"],
        "class_index_a": ia, "class_index_b": ib,
        "n_photos": len(photos_a), "n_joint_positive": sum(joint),
        "n_gt": sum(len(gm) for gm, _ in stats_a),
        "recall_a": round(ra, 4), "recall_b": round(rb, 4),
        "threshold_a": ta, "threshold_b": tb,
        "operating_point_a": op_a, "operating_point_b": op_b,
        "escape_a": None if ea is None else round(ea, 4),
        "escape_b": None if eb is None else round(eb, 4),
        "seed": seed,
    })
    if margin is not None:
        lo = rc[0] if rc else None
        result["margin"] = margin
        result["non_inferiority"] = (
            f"A non-inferior to B (CI lower bound {lo} > -{margin})"
            if lo is not None and lo > -margin
            else f"non-inferiority NOT shown (CI lower bound {lo} <= -{margin})"
        )
        result["non_inferior"] = bool(lo is not None and lo > -margin)
    if write:
        out = settings.eval_dir / f"compare_{A['name']}__{B['name']}__{class_slug(cname)}.json"
        out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
        result["file"] = str(out)
    logger.info(json.dumps(result, indent=1, ensure_ascii=False))
    return result


def run(run_a: str, run_b: str, class_index: int = 0, seed: int = 0) -> dict:
    """Frozen API (CONTRACTS §8.2): compare two runs/families on run A's class
    `class_index` (B must have the same class)."""
    return compare(run_a, run_b, cls=class_index, seed=seed)


def summary_line(r: dict) -> str:
    """One human line (\"n/a\" for missing operating points)."""
    esc = "n/a" if r["escape_diff"] is None else f"{r['escape_diff']:+.3f} (CI {r['escape_diff_ci95']})"
    line = (f"{r['run_a']} vs {r['run_b']} [{r['class']}, {r['bucket']}]: recall@0.5FPPI "
            f"{r['recall_a']:.3f} vs {r['recall_b']:.3f}, diff {r['recall_at_0.5fppi_diff']:+.3f} "
            f"(CI {r['recall_diff_ci95']}) -> {r['g1_decision']}; escape diff {esc} "
            f"(ops: {r['operating_point_a']} / {r['operating_point_b']})")
    if "non_inferiority" in r:
        line += f"; {r['non_inferiority']}"
    return line
