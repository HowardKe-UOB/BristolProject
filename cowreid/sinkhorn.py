"""Entropic optimal transport (Sinkhorn) in pure NumPy.

Used by Tier-2 mining to softly match the bag of crops seen in camera A at time t
against the bag seen in camera B at the same instant. Bags are tiny (<= ~12), so a
plain log-domain Sinkhorn is fast and numerically stable. No POT/scipy dependency.
"""
from __future__ import annotations

import numpy as np


def sinkhorn(cost: np.ndarray, a: np.ndarray | None = None, b: np.ndarray | None = None,
             eps: float = 0.1, n_iters: int = 200, tol: float = 1e-6) -> np.ndarray:
    """Return the (n, m) transport plan minimising <P, cost> - eps * H(P).

    ``a`` / ``b`` are the source/target marginals (default: uniform). Computed in
    log domain for stability.
    """
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    if a is None:
        a = np.full(n, 1.0 / n)
    if b is None:
        b = np.full(m, 1.0 / m)
    log_a = np.log(np.asarray(a, dtype=np.float64) + 1e-300)
    log_b = np.log(np.asarray(b, dtype=np.float64) + 1e-300)

    K = -cost / eps                      # log-kernel
    f = np.zeros(n)
    g = np.zeros(m)
    for _ in range(n_iters):
        f_prev = f
        # f_i = log a_i - logsumexp_j (K_ij + g_j)
        f = log_a - _logsumexp(K + g[None, :], axis=1)
        g = log_b - _logsumexp(K + f[:, None], axis=0)
        if np.max(np.abs(f - f_prev)) < tol:
            break
    log_P = f[:, None] + K + g[None, :]
    return np.exp(log_P)


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    mx = np.max(x, axis=axis, keepdims=True)
    out = mx + np.log(np.sum(np.exp(x - mx), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def match_with_dustbin(cost: np.ndarray, eps: float = 0.1,
                       dustbin_cost: float | None = None,
                       dustbin_quantile: float = 0.5
                       ) -> list[tuple[int, int, float]]:
    """OT matching with a reject option.

    On this dataset the true cross-view partner is *absent* from the other
    camera's bag ~90% of the time (cameras only partially overlap), so a matcher
    that is forced to transport all mass produces mostly wrong matches. We augment
    the cost with a dustbin row/column at ``dustbin_cost`` (default: the
    ``dustbin_quantile`` quantile of the real costs) into which unmatched mass
    flows. A pair (i, j) is returned only if row i and column j prefer each other
    over the dustbin *and* over all alternatives.

    Returns ``(i, j, confidence)`` with confidence = row mass on j / row total.
    """
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    if n == 0 or m == 0:
        return []
    if dustbin_cost is None:
        dustbin_cost = float(np.quantile(cost, dustbin_quantile))

    aug = np.full((n + 1, m + 1), dustbin_cost, dtype=np.float64)
    aug[:n, :m] = cost
    aug[n, m] = 0.0
    a = np.full(n + 1, 1.0); a[n] = float(m)
    b = np.full(m + 1, 1.0); b[m] = float(n)
    a /= a.sum(); b /= b.sum()

    P = sinkhorn(aug, a, b, eps=eps)
    row_sum = P[:n, :].sum(axis=1) + 1e-300
    out = []
    for i in range(n):
        j = int(P[i, :m].argmax())
        # prefer a real column over the dustbin, and be column i's best real row
        if P[i, j] <= P[i, m]:
            continue
        if int(P[:n, j].argmax()) != i:
            continue
        out.append((i, j, float(P[i, j] / row_sum[i])))
    return out


def mutual_matches(P: np.ndarray) -> list[tuple[int, int, float]]:
    """Mutual-nearest-neighbour matches in a transport plan.

    Returns ``(i, j, row_concentration)`` where row_concentration in (0, 1] is the
    fraction of row i's transported mass placed on column j (higher = more
    confident). A pair is returned only if i is j's best row AND j is i's best col.
    """
    if P.size == 0:
        return []
    row_best = P.argmax(axis=1)
    col_best = P.argmax(axis=0)
    row_sum = P.sum(axis=1) + 1e-300
    out = []
    for i, j in enumerate(row_best):
        if col_best[j] == i:
            out.append((int(i), int(j), float(P[i, j] / row_sum[i])))
    return out
