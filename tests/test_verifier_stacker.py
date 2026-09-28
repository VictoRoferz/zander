"""
The numpy stacker (ml/verifier/stacker.py): Newton/IRLS logistic regression
with L2 and an unpenalised intercept, the tau rule, hi-passthrough and the
rank AUROC.

tau fixture (t = 0.3), by hand:
  A positive, detector 0.8 (flagged), stacked p {0.6, 0.2}        -> needs tau <= 0.6
  B positive, detector 0.5 (flagged), stacked p {0.35}            -> needs tau <= 0.35
  C positive, detector 0.25 (NOT flagged at t)                    -> no constraint
  D positive, flagged, a passthrough candidate at 0.9 >= t        -> covered
  E iO photo (negative), stacked p 0.99                           -> ignored
  F positive, flagged, stacked (p 0.05, det 0.7, hi 0.65): det >= hi and >= t -> covered
  => tau = min(0.6, 0.35) = 0.35 (inclusive: B stays flagged at p = tau), binding photo B.

Run (no pytest needed):  ml/.venv/bin/python -B tests/test_verifier_stacker.py
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))

import numpy as np  # noqa: E402

from infer.verifier import stack_proba  # noqa: E402
from verifier.stacker import (  # noqa: E402
    TAU_MAX, auroc, choose_hi_passthrough, fit_class_stacker, fit_logistic, select_tau,
)


def synthetic(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    w, b = np.array([1.5, -2.0, 0.5]), -0.7
    y = (rng.random(n) < 1 / (1 + np.exp(-(b + x @ w)))).astype(float)
    return x, y, w, b


def test_irls_recovers_known_coefficients():
    x, y, w, b = synthetic()
    b_hat, w_hat, info = fit_logistic(x, y, C=1e6)
    assert info["converged"], info
    assert np.max(np.abs(w_hat - w)) < 0.08, (w_hat, w)
    assert abs(b_hat - b) < 0.08, (b_hat, b)
    # C = 1 on 20k rows: the penalty is negligible
    b1, w1, _ = fit_logistic(x, y, C=1.0)
    assert np.max(np.abs(w1 - w_hat)) < 5e-3 and abs(b1 - b_hat) < 5e-3


def test_irls_matches_sklearn_objective_when_available():
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        print("  (skip: scikit-learn not installed)")
        return
    x, y, _, _ = synthetic(n=3000, seed=1)
    b_hat, w_hat, _ = fit_logistic(x, y, C=0.05)
    sk = LogisticRegression(C=0.05, tol=1e-10, max_iter=10000).fit(x, y)  # intercept unpenalised
    assert np.max(np.abs(sk.coef_[0] - w_hat)) < 1e-4 and abs(sk.intercept_[0] - b_hat) < 1e-4


def test_standardised_wrapper_reproduces_raw_scale_probabilities():
    x, y, w, b = synthetic(n=8000, seed=2)
    x_raw = x * np.array([10.0, 0.1, 3.0]) + np.array([5.0, -1.0, 0.0])
    params, info = fit_class_stacker(np.hstack([x_raw, np.ones((len(x), 1))]), y, C=1e6)
    assert params["std"][3] == 1.0  # a constant feature keeps std 1 (no division by 0)
    p_true = 1 / (1 + np.exp(-(b + x @ w)))
    p_hat = stack_proba(params, np.hstack([x_raw, np.ones((len(x), 1))]))
    err = np.abs(p_hat - p_true)  # sampling noise of 8k rows, not a fit error
    assert err.mean() < 0.015 and err.max() < 0.08, (err.mean(), err.max())
    assert abs(params['coef'][0] - 1.5) < 0.15 and abs(params['coef'][1] + 2.0) < 0.15


def test_separable_data_stays_finite():
    x = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    b, w, info = fit_logistic(x, y, C=1.0)
    assert info["converged"] and np.all(np.isfinite(w)) and math.isfinite(b)
    assert 0 < w[0] < 10 and abs(b) < 1e-6  # symmetric data -> intercept 0
    p = 1 / (1 + np.exp(-(b + x[:, 0] * w[0])))
    assert np.all((p > 0) & (p < 1)) and np.all(p[3:] > 0.5) and np.all(p[:3] < 0.5)
    # perfectly separable in 4-d as well
    rng = np.random.default_rng(3)
    x4 = rng.normal(size=(200, 4))
    y4 = (x4[:, 0] > 0).astype(float)
    b4, w4, info4 = fit_logistic(x4, y4, C=1.0)
    assert info4["converged"] and np.all(np.isfinite(w4)) and np.max(np.abs(w4)) < 50


def test_single_class_is_refused():
    for y in (np.zeros(5), np.ones(5)):
        try:
            fit_logistic(np.ones((5, 2)), y)
            raise AssertionError("a one-class fit must be refused")
        except ValueError:
            pass


def test_tau_hand_worked_fixture():
    photos = [
        {"capture_id": "A", "positive": True, "det_photo_score": 0.8,
         "stacked": [(0.6, 0.8), (0.2, 0.1)], "passthrough": []},
        {"capture_id": "B", "positive": True, "det_photo_score": 0.5,
         "stacked": [(0.35, 0.5)], "passthrough": []},
        {"capture_id": "C", "positive": True, "det_photo_score": 0.25,
         "stacked": [(0.05, 0.25)], "passthrough": []},
        {"capture_id": "D", "positive": True, "det_photo_score": 0.9,
         "stacked": [(0.01, 0.2)], "passthrough": [0.9]},
        {"capture_id": "E", "positive": False, "det_photo_score": 0.95,
         "stacked": [(0.99, 0.95)], "passthrough": []},
        {"capture_id": "F", "positive": True, "det_photo_score": 0.7,
         "stacked": [(0.05, 0.7, 0.65)], "passthrough": []},
    ]
    tau, info = select_tau(photos, t=0.3)
    assert tau == 0.35, tau
    assert info["positives_flagged_by_detector"] == 4  # A, B, D, F
    assert info["covered_without_stacker"] == 2  # D (passthrough), F (hi)
    assert info["constraints"] == 2 and info["binding_photo"] == "B"
    # every detector-flagged positive stays flagged at tau (inclusive)
    for ph in photos:
        if ph["positive"] and ph["det_photo_score"] >= 0.3:
            ok = (any(s >= 0.3 for s in ph["passthrough"])
                  or any(len(e) > 2 and e[1] >= e[2] for e in ph["stacked"])
                  or max(e[0] for e in ph["stacked"]) >= tau)
            assert ok, ph["capture_id"]
    # a global hi of 0.75 covers A as well -> only B constrains
    tau2, info2 = select_tau(photos, t=0.3, hi=0.75)
    assert tau2 == 0.35 and info2["covered_without_stacker"] == 3
    # nothing to keep -> the cap, flagged as unconstrained
    tau3, info3 = select_tau(photos[4:5], t=0.3)
    assert tau3 == TAU_MAX and info3["unconstrained"]


def test_hi_passthrough_auto_rule():
    det = [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.88]
    lab = [1, 1, 1, 1, 1, 0, 1, -1]  # the -1 (ignored) never breaks the pure region
    assert choose_hi_passthrough(det, lab, t=0.3, min_pos=5) == 0.7
    assert choose_hi_passthrough(det, lab, t=0.3, min_pos=6) is None
    assert choose_hi_passthrough(det, lab, t=0.75, min_pos=5) == 0.75  # never below t
    assert choose_hi_passthrough([0.9, 0.8], [1, 1], t=0.1, min_pos=2) == 0.8  # no negatives


def test_rank_auroc():
    assert auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == 0.75
    assert auroc([0.5, 0.5], [1, 0]) == 0.5
    assert auroc([0.1, 0.9, 0.3], [0, 1, -1]) == 1.0  # label -1 ignored
    assert auroc([0.1, 0.2], [1, 1]) is None
    rng = np.random.default_rng(4)
    s, lab = rng.random(300), rng.integers(0, 2, 300)
    pairs = [(a, b) for a in s[lab == 1] for b in s[lab == 0]]
    brute = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a, b in pairs) / len(pairs)
    assert abs(auroc(s, lab) - brute) < 1e-12


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
