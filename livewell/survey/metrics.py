from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import stats

Array = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]


def _midrank(x: Array) -> Array:
    order = np.argsort(x)
    ranked = x[order]
    n = len(x)
    out = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and ranked[j] == ranked[i]:
            j += 1
        out[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    result = np.empty(n, dtype=np.float64)
    result[order] = out
    return result


def auroc(scores: Array, labels: IntArray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    ranks = _midrank(np.concatenate([pos, neg]))
    rank_pos = ranks[: pos.size].sum()
    return float((rank_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def _delong_components(scores: Array, labels: IntArray) -> tuple[float, Array, Array]:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    m = pos.size
    n = neg.size
    tx = _midrank(pos)
    ty = _midrank(neg)
    tz = _midrank(np.concatenate([pos, neg]))
    auc = (tz[:m].sum() - m * (m + 1) / 2.0) / (m * n)
    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m
    return float(auc), v01, v10


def delong_test(scores_a: Array, scores_b: Array, labels: IntArray) -> tuple[float, float, float]:
    auc_a, a01, a10 = _delong_components(scores_a, labels)
    auc_b, b01, b10 = _delong_components(scores_b, labels)
    m = a01.size
    n = a10.size
    cov01 = np.cov(np.vstack([a01, b01]))
    cov10 = np.cov(np.vstack([a10, b10]))
    cov = cov01 / m + cov10 / n
    var = cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]
    if var <= 0.0:
        return auc_a, auc_b, 1.0
    z = (auc_a - auc_b) / np.sqrt(var)
    p = 2.0 * stats.norm.sf(abs(z))
    return auc_a, auc_b, float(p)


def cohen_kappa(a: IntArray, b: IntArray) -> float:
    observed = float(np.mean(a == b))
    classes = np.unique(np.concatenate([a, b]))
    expected = 0.0
    for c in classes:
        expected += (np.mean(a == c)) * (np.mean(b == c))
    if expected >= 1.0:
        return 1.0
    return float((observed - expected) / (1.0 - expected))


def balanced_accuracy(truth: IntArray, pred: IntArray) -> float:
    sens = float(np.mean(pred[truth == 1] == 1)) if np.any(truth == 1) else 0.0
    spec = float(np.mean(pred[truth == 0] == 0)) if np.any(truth == 0) else 0.0
    return 0.5 * (sens + spec)


def lin_ccc(x: Array, y: Array) -> float:
    mx = float(np.mean(x))
    my = float(np.mean(y))
    vx = float(np.var(x))
    vy = float(np.var(y))
    cov = float(np.mean((x - mx) * (y - my)))
    denom = vx + vy + (mx - my) ** 2
    if denom <= 0.0:
        return 0.0
    return 2.0 * cov / denom
