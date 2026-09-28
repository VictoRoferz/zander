"""
The logistic stacker and its thresholds — numpy only.

fit_logistic: Newton / IRLS for
    min  sum_i [log(1 + e^{z_i}) - y_i z_i] + ||w||^2 / (2C),   z = b + X w
on standardised features, intercept b UNpenalised (sklearn's C convention).
Backtracking line search; the L2 term keeps w finite on separable data, and b
is finite whenever both classes are present (one class only -> ValueError).

select_tau: the highest tau at which every selection-fold POSITIVE photo that
the detector flags at t is still flagged by the cascade — a stacked candidate
with p >= tau, a passthrough candidate with s_det >= t, or a hi-passthrough
candidate (s_det >= h and s_det >= t). So on the selection folds the cascade
adds no photo escape relative to the detector at t, by construction; with
several stacked classes tau is shared (p is a calibrated probability for
every class).

choose_hi_passthrough ("auto"): the lowest detector score above which every
fitted candidate of the class is a true positive, if at least `min_pos`
positives lie there (else None = off); never below t.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np

from infer.verifier import sigmoid

TAU_MIN, TAU_MAX = 1e-6, 1.0 - 1e-6


def fit_logistic(X, y, C: float = 1.0, max_iter: int = 100,
                 tol: float = 1e-10) -> tuple[float, np.ndarray, dict]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X must be [n, d] and aligned with y")
    n, d = X.shape
    if n == 0 or not ((y == 0) | (y == 1)).all():
        raise ValueError("y must be a non-empty 0/1 vector")
    if y.min() == y.max():
        raise ValueError("logistic fit needs both classes (all labels are %d)" % y[0])
    a = np.hstack([np.ones((n, 1)), X])
    lam = np.full(d + 1, 1.0 / C)
    lam[0] = 0.0
    theta = np.zeros(d + 1)
    ybar = float(y.mean())
    theta[0] = math.log(ybar / (1.0 - ybar))

    def objective(th):
        z = a @ th
        return float(np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * np.sum(lam * th * th))

    f = objective(theta)
    converged, it = False, 0
    for it in range(1, max_iter + 1):
        z = a @ theta
        p = np.asarray(sigmoid(z)).reshape(-1)
        g = a.T @ (p - y) + lam * theta
        w = p * (1.0 - p)
        h = a.T @ (a * w[:, None]) + np.diag(lam) + 1e-10 * np.eye(d + 1)
        step = np.linalg.solve(h, g)
        t = 1.0
        while True:
            cand = theta - t * step
            fc = objective(cand)
            if fc <= f - 1e-4 * t * float(g @ step) or t < 1e-12:
                break
            t *= 0.5
        delta = float(np.max(np.abs(cand - theta)))
        theta, f = cand, fc
        if delta < tol:
            converged = True
            break
    if not np.all(np.isfinite(theta)):
        raise FloatingPointError("logistic fit diverged")
    return float(theta[0]), theta[1:].copy(), {
        "iterations": it, "converged": converged, "objective": round(f, 6), "n": n,
        "n_pos": int(y.sum()),
    }


def fit_class_stacker(x_raw, y, C: float = 1.0) -> tuple[dict, dict]:
    """Standardise, fit, and return the stacker.json class params (without
    tau / hi_passthrough) plus fit info."""
    x_raw = np.asarray(x_raw, dtype=np.float64)
    mean = x_raw.mean(axis=0)
    std = x_raw.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    b, w, info = fit_logistic((x_raw - mean) / std, y, C=C)
    return {"mean": mean.tolist(), "std": std.tolist(), "coef": w.tolist(),
            "intercept": b}, info


def choose_hi_passthrough(det_scores, labels, t: float, min_pos: int = 5) -> Optional[float]:
    det = np.asarray(det_scores, dtype=np.float64)
    lab = np.asarray(labels)
    neg = det[lab == 0]
    top_neg = float(neg.max()) if len(neg) else -math.inf
    pure = det[(lab == 1) & (det > top_neg)]
    if len(pure) < min_pos:
        return None
    return float(max(pure.min(), t))


def resolve_hi(setting, det_scores, labels, t: float, min_pos: int) -> Optional[float]:
    if setting is None:
        return None
    if setting == "auto":
        return choose_hi_passthrough(det_scores, labels, t, min_pos)
    return float(setting)


def _hi_covers(entry, t: float, hi: Optional[float]) -> bool:
    h = entry[2] if len(entry) > 2 else hi
    return h is not None and entry[1] >= h and entry[1] >= t


def select_tau(photos: Iterable[dict], t: float, hi: Optional[float] = None) -> tuple[float, dict]:
    """photos: {positive: bool, det_photo_score: float,
                stacked: [(p, s_det) or (p, s_det, hi_of_its_class), ...],
                passthrough: [s_det, ...]}."""
    constraints, n_pos_flagged, n_covered = [], 0, 0
    binding = None
    for ph in photos:
        if not ph["positive"] or ph["det_photo_score"] < t:
            continue
        n_pos_flagged += 1
        if any(s >= t for s in ph["passthrough"]) or any(
            _hi_covers(e, t, hi) for e in ph["stacked"]
        ):
            n_covered += 1
            continue
        if not ph["stacked"]:
            raise ValueError(
                f"photo {ph.get('capture_id')} is flagged by the detector at t but has no "
                "candidate that could flag it (top-k/label mismatch?)"
            )
        m = max(e[0] for e in ph["stacked"])
        constraints.append(m)
        if binding is None or m < binding[1]:
            binding = (ph.get("capture_id"), m)
    tau = min(constraints) if constraints else TAU_MAX
    clipped = min(max(tau, TAU_MIN), TAU_MAX)
    return float(clipped), {
        "t": t, "hi_passthrough": hi, "positives_flagged_by_detector": n_pos_flagged,
        "covered_without_stacker": n_covered, "constraints": len(constraints),
        "binding_photo": binding[0] if binding else None,
        "unconstrained": not constraints, "clipped": clipped != tau,
    }


def auroc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """Mann-Whitney AUROC via mid-ranks (ties count 1/2); labels 1/0, others ignored."""
    s = np.asarray(scores, dtype=np.float64)
    lab = np.asarray(labels)
    keep = (lab == 0) | (lab == 1)
    s, lab = s[keep], lab[keep]
    n_pos, n_neg = int((lab == 1).sum()), int((lab == 0).sum())
    if not n_pos or not n_neg:
        return None
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    mid_rank = np.cumsum(counts) - (counts - 1) / 2.0  # 1-based average rank per value
    r_pos = mid_rank[inv][lab == 1].sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
