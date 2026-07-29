"""Small linear-algebra + GLM utilities shared by base_model.py and the ban-
model agent_strength extension. No numpy/scipy dependency (consistent with
veto_model.py) -- feature counts here are always small (1-3), so plain-Python
Gauss-Jordan and Newton-Raphson are simple and exact enough.
"""

from __future__ import annotations

import math


def matrix_inverse(m: list[list[float]]) -> list[list[float]]:
    """Gauss-Jordan inverse of a small square matrix."""
    n = len(m)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(m)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-12:
            raise ValueError("matrix is singular (or near-singular) -- cannot invert")
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot = aug[col][col]
        aug[col] = [v / pivot for v in aug[col]]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor != 0:
                aug[r] = [aug[r][c] - factor * aug[col][c] for c in range(2 * n)]
    return [row[n:] for row in aug]


def matvec(m: list[list[float]], v: list[float]) -> list[float]:
    return [sum(m[i][j] * v[j] for j in range(len(v))) for i in range(len(m))]


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def fit_logistic_no_intercept(X: list[list[float]], y: list[int], max_iters: int = 100, tol: float = 1e-9) -> dict:
    """Newton-Raphson (IRLS) fit of a no-intercept logistic regression.
    X is a list of feature vectors (each length k), y is 0/1 labels.
    Returns beta, the Fisher information matrix at convergence (X'WX, the
    "bread" for a robust sandwich variance), and per-observation scores
    x_i*(y_i - p_i) at the final beta (the ingredients for the "meat").
    """
    n = len(y)
    k = len(X[0]) if X else 0
    beta = [0.0] * k
    converged = False
    info = [[0.0] * k for _ in range(k)]

    for _ in range(max_iters):
        p = [sigmoid(sum(X[i][j] * beta[j] for j in range(k))) for i in range(n)]
        w = [max(p[i] * (1 - p[i]), 1e-10) for i in range(n)]
        grad = [sum(X[i][j] * (y[i] - p[i]) for i in range(n)) for j in range(k)]
        info = [[sum(X[i][a] * w[i] * X[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
        try:
            info_inv = matrix_inverse(info)
        except ValueError:
            break
        step = matvec(info_inv, grad)
        beta = [beta[j] + step[j] for j in range(k)]
        if max(abs(s) for s in step) < tol:
            converged = True
            break

    p_final = [sigmoid(sum(X[i][j] * beta[j] for j in range(k))) for i in range(n)]
    scores = [[X[i][j] * (y[i] - p_final[i]) for j in range(k)] for i in range(n)]
    log_likelihood = sum(math.log(max(p_final[i] if y[i] == 1 else 1 - p_final[i], 1e-12)) for i in range(n))

    return {
        "beta": beta,
        "info_matrix": info,
        "scores": scores,  # per-observation, for cluster-robust SE
        "converged": converged,
        "log_likelihood": log_likelihood,
    }


def cluster_robust_se(info_matrix: list[list[float]], scores: list[list[float]], cluster_ids: list) -> list[float]:
    """Cluster-robust ("sandwich") standard errors for a fitted GLM.

    bread = (X'WX)^-1 (info_matrix, already inverted here)
    meat  = sum over clusters c of (sum_{i in c} score_i) (sum_{i in c} score_i)'
    Var   = bread * meat * bread

    Clustering on match_id because observations from the same match (multiple
    maps, multiple bans/picks) are correlated -- naive (non-clustered)
    standard errors would be too small, exactly the failure mode this
    project has already been warned about.
    """
    k = len(info_matrix)
    bread = matrix_inverse(info_matrix)

    cluster_sums: dict = {}
    for cid, score in zip(cluster_ids, scores):
        acc = cluster_sums.setdefault(cid, [0.0] * k)
        for j in range(k):
            acc[j] += score[j]

    meat = [[0.0] * k for _ in range(k)]
    for acc in cluster_sums.values():
        for a in range(k):
            for b in range(k):
                meat[a][b] += acc[a] * acc[b]

    bread_meat = [[sum(bread[a][c] * meat[c][b] for c in range(k)) for b in range(k)] for a in range(k)]
    var = [[sum(bread_meat[a][c] * bread[b][c] for c in range(k)) for b in range(k)] for a in range(k)]
    return [math.sqrt(max(var[j][j], 0.0)) for j in range(k)]


def log_loss(X: list[list[float]], y: list[int], beta: list[float]) -> float:
    """Mean negative log-likelihood (nats/observation) -- lower is better."""
    n = len(y)
    if n == 0:
        return float("nan")
    total = 0.0
    for i in range(n):
        p = sigmoid(sum(X[i][j] * beta[j] for j in range(len(beta))))
        total -= math.log(max(p if y[i] == 1 else 1 - p, 1e-12))
    return total / n


def perplexity(nats_per_obs: float) -> float:
    """exp(nats/obs): the effective number of equally-likely outcomes implied
    by the average log-loss -- e.g. 2.0 means "as uncertain as a fair coin
    flip", 1.0 means perfect certainty."""
    return math.exp(nats_per_obs)
