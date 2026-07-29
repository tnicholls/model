"""Test 2: does adding agent_strength as a third feature to the existing
conditional-logit ban model (veto_model.py) improve held-out log loss?
Expectation per the build spec is null -- this is a cheap re-run on data
that already exists from base_model.build_walkforward's veto_decisions.

Reuses base_model.py's (kappa_player, kappa_agent_map) tuned for Model 1
rather than re-tuning them here -- the spec frames this test as "cheap,
expected null", not worth its own hyperparameter search. own/opp map-rate
shrinkage reuses kappa=0.3, the value the existing ban model (veto_model.py)
already found and validated on this same decision data.

The 2- and 3-feature softmax-over-pool models are both instances of a single
N-feature conditional logit, generalizing veto_model.fit_model (hardcoded to
2 features) with the same gradient-ascent-plus-backtracking-line-search
optimizer.
"""

from __future__ import annotations

import math

from .veto_model import shrunk_rate
from .base_model import _agent_strength_value

EXISTING_BAN_MODEL_KAPPA = 0.3  # veto_model.run_likelihood_ratio_test's found interior optimum for bans


def _softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    s = sum(exps)
    return [e / s for e in exps]


def _build_features_n(decisions: list[dict], kappa_map: float, kappa_player: float, kappa_agent_map: float, n_features: int) -> list[tuple[int, list[list[float]]]]:
    """(chosen_index, [[f1,f2,...] per pool map]) per decision. n_features=2
    -> own/opp map-rate only (reproduces veto_model's existing feature set
    exactly). n_features=3 -> + agent_strength diff (own - opp) per map."""
    out = []
    for d in decisions:
        pool = d["pool"]
        chosen_idx = pool.index(d["chosen"])
        rows = []
        for mp in pool:
            own_rate = -shrunk_rate(d["own_counts"][mp], d["own_global"], kappa_map)
            opp_rate = shrunk_rate(d["opp_counts"][mp], d["opp_global"], kappa_map)
            feats = [own_rate, opp_rate]
            if n_features == 3:
                own_terms = d["own_agent_terms_by_map"].get(mp, [])
                opp_terms = d["opp_agent_terms_by_map"].get(mp, [])
                own_strength = _agent_strength_value(own_terms, kappa_player, kappa_agent_map)
                opp_strength = _agent_strength_value(opp_terms, kappa_player, kappa_agent_map)
                feats.append(own_strength - opp_strength)
            rows.append(feats)
        out.append((chosen_idx, rows))
    return out


def _log_likelihood(features: list[tuple[int, list[list[float]]]], beta: list[float]) -> float:
    ll = 0.0
    for chosen_idx, rows in features:
        logits = [sum(b * f for b, f in zip(beta, row)) for row in rows]
        p = _softmax(logits)
        ll += math.log(max(p[chosen_idx], 1e-12))
    return ll


def _fit_n(features: list[tuple[int, list[list[float]]]], n_features: int, max_iters: int = 5000, grad_tol: float = 1e-7) -> dict:
    beta = [0.0] * n_features
    ll = _log_likelihood(features, beta)
    step = 1.0
    converged = False
    iterations = 0

    for iterations in range(1, max_iters + 1):
        grad = [0.0] * n_features
        for chosen_idx, rows in features:
            logits = [sum(b * f for b, f in zip(beta, row)) for row in rows]
            p = _softmax(logits)
            for j in range(n_features):
                e_fj = sum(p[i] * rows[i][j] for i in range(len(rows)))
                grad[j] += rows[chosen_idx][j] - e_fj
        grad = [g / len(features) for g in grad]
        grad_norm = math.sqrt(sum(g * g for g in grad))
        if grad_norm < grad_tol:
            converged = True
            break

        trial = step
        moved = False
        while trial > 1e-12:
            candidate = [beta[j] + trial * grad[j] for j in range(n_features)]
            candidate_ll = _log_likelihood(features, candidate)
            if candidate_ll >= ll:
                beta, ll = candidate, candidate_ll
                step = trial * 1.2
                moved = True
                break
            trial *= 0.5
        if not moved:
            converged = True
            break

    return {"beta": beta, "log_likelihood": ll, "converged": converged, "iterations": iterations}


def _null_log_likelihood(decisions: list[dict]) -> float:
    return sum(-math.log(len(d["pool"])) for d in decisions)


def _chronological_split(decisions: list[dict], test_frac: float = 0.2, tune_frac: float = 0.2) -> tuple[list[dict], list[dict], list[dict]]:
    n = len(decisions)
    n_test = max(10, int(n * test_frac))
    n_tune = max(10, int(n * tune_frac))
    return decisions[: n - n_test - n_tune], decisions[n - n_test - n_tune : n - n_test], decisions[n - n_test :]


def run(veto_decisions: list[dict], kappa_player: float, kappa_agent_map: float, action_types: tuple[str, ...] = ("ban",), min_history_games: int = 3) -> dict:
    """Held-out log loss (nats/decision) for the existing 2-feature ban model
    vs. the +agent_strength 3-feature extension, overall and on the
    pre-specified stale subset. Same eligibility gate (min_history_games on
    global game counts) and same chronological fit/tune/test split as
    veto_model.run_likelihood_ratio_test.
    """
    decisions = [d for d in veto_decisions if d["action"] in action_types]
    decisions = [d for d in decisions if sum(d["own_global"]) >= min_history_games and sum(d["opp_global"]) >= min_history_games]
    if len(decisions) < 30:
        raise ValueError(f"only {len(decisions)} eligible decisions -- too few to fit/test")

    fit_only, tune, test = _chronological_split(decisions)
    fit_plus_tune = fit_only + tune

    results = {}
    for n_features, key in ((2, "existing_2feature"), (3, "extended_3feature")):
        feats_fit = _build_features_n(fit_plus_tune, EXISTING_BAN_MODEL_KAPPA, kappa_player, kappa_agent_map, n_features)
        fit = _fit_n(feats_fit, n_features)

        def subset_ll(subset: list[dict]) -> tuple[float | None, int]:
            """Mean NEGATIVE log-likelihood (nats/decision, positive, lower is
            better) -- matches base_model/stats_utils' log_loss convention."""
            if not subset:
                return None, 0
            feats = _build_features_n(subset, EXISTING_BAN_MODEL_KAPPA, kappa_player, kappa_agent_map, n_features)
            return -_log_likelihood(feats, fit["beta"]) / len(subset), len(subset)

        overall_ll, n_overall = subset_ll(test)
        stale_test = [d for d in test if d["stale"] is True]
        nonstale_test = [d for d in test if d["stale"] is False]
        stale_ll, n_stale = subset_ll(stale_test)
        nonstale_ll, n_nonstale = subset_ll(nonstale_test)

        results[key] = {
            "beta": fit["beta"],
            "converged": fit["converged"],
            "n_test": n_overall,
            "log_loss_nats": overall_ll,
            "perplexity": math.exp(overall_ll) if overall_ll is not None else None,
            "n_stale": n_stale,
            "log_loss_nats_stale": stale_ll,
            "perplexity_stale": math.exp(stale_ll) if stale_ll is not None else None,
            "n_nonstale": n_nonstale,
            "log_loss_nats_nonstale": nonstale_ll,
            "perplexity_nonstale": math.exp(nonstale_ll) if nonstale_ll is not None else None,
        }

    return {
        "action_types": action_types,
        "n_eligible": len(decisions),
        "n_fit": len(fit_only),
        "n_tune": len(tune),
        "n_test": len(test),
        "kappa_map": EXISTING_BAN_MODEL_KAPPA,
        "kappa_player": kappa_player,
        "kappa_agent_map": kappa_agent_map,
        **results,
        "improvement_nats_overall": (
            results["existing_2feature"]["log_loss_nats"] - results["extended_3feature"]["log_loss_nats"]
            if results["existing_2feature"]["log_loss_nats"] is not None and results["extended_3feature"]["log_loss_nats"] is not None
            else None
        ),
        "improvement_nats_stale": (
            results["existing_2feature"]["log_loss_nats_stale"] - results["extended_3feature"]["log_loss_nats_stale"]
            if results["existing_2feature"]["log_loss_nats_stale"] is not None and results["extended_3feature"]["log_loss_nats_stale"] is not None
            else None
        ),
    }
