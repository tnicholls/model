"""Shrinkage win-rate estimation + a single likelihood-ratio test for the veto
model, built on top of the raw pick/ban data collected by veto.py.

Two upgrades over the earlier accuracy/Bonferroni approach in
veto.test_ban_rules:

  * Empirical-Bayes shrinkage. A team's map win rate is pulled toward its
    overall (all-map) win rate, weighted by how many games it's actually
    played on that map:

        p_hat = (wins + kappa * team_global_rate) / (n_games + kappa)

    kappa controls how hard to shrink -- kappa=0 is the raw (noisy) rate,
    kappa=infinity ignores the map entirely and just uses team strength.
    It's chosen by holding out a slice of the data, not guessed.

  * One likelihood-ratio test instead of many corrected accuracy tests. A
    2-parameter softmax-over-available-pool model -- own weakness and
    opponent strength as the two features -- is fit by maximum likelihood
    on ban decisions and compared against the uniform-over-pool null via a
    likelihood-ratio test. One p-value, uses every decision, no
    multiple-comparisons patchwork.

Everything here operates on a single walk-forward pass (collect_decisions)
that records raw win/loss counts *as of the moment of each decision* --
kappa tuning and rule variants can then be explored cheaply in memory
without re-walking the (much more expensive to build) raw dataset again.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict

from .veto import TEAM_TAGS_PATH, VETO_DATASET_PATH, VetoStep, _resolve_veto_team


def _load_rows() -> tuple[list[dict], dict]:
    tag_cache = json.loads(TEAM_TAGS_PATH.read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in VETO_DATASET_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows.sort(key=lambda r: r["match_id"])  # match_id order == chronological proxy
    return rows, tag_cache


def collect_decisions(action_types: tuple[str, ...] = ("ban", "pick")) -> list[dict]:
    """One walk-forward pass over the dataset. For every ban/pick decision of
    a requested action type, records the acting/opponent team's raw
    (unshrunk) per-map and global win/loss counts *before* that decision,
    plus the available pool and which map was actually chosen. History is
    updated with a match's real results only after all of that match's
    decisions have been recorded -- no lookahead.
    """
    rows, tag_cache = _load_rows()

    record: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    global_record: dict[int, list[int]] = defaultdict(lambda: [0, 0])

    decisions: list[dict] = []

    for row in rows:
        t1_id, t2_id = row.get("team1_id"), row.get("team2_id")
        veto = [VetoStep(**s) for s in row["veto"]]
        available = set(s.map for s in veto)
        position: dict[str, int] = defaultdict(int)

        for step in veto:
            if step.action not in ("ban", "pick"):
                continue
            position[step.action] += 1

            if step.action in action_types:
                team_id = _resolve_veto_team(step.team, t1_id, t2_id, tag_cache)
                if team_id is not None and len(available) >= 2:
                    opp_id = t2_id if team_id == t1_id else t1_id
                    if opp_id is not None:
                        decisions.append(
                            {
                                "match_id": row["match_id"],
                                "action": step.action,
                                "position": position[step.action],
                                "pool": sorted(available),
                                "chosen": step.map,
                                "own_counts": {m: list(record[team_id][m]) for m in available},
                                "opp_counts": {m: list(record[opp_id][m]) for m in available},
                                "own_global": list(global_record[team_id]),
                                "opp_global": list(global_record[opp_id]),
                            }
                        )
            available.discard(step.map)

        team_id_by_name = {row["team1"]: t1_id, row["team2"]: t2_id}
        for mr in row["maps"]:
            if mr["winner"] is None:
                continue
            loser_name = mr["team1"] if mr["winner"] == mr["team2"] else mr["team2"]
            winner_id = team_id_by_name.get(mr["winner"])
            loser_id = team_id_by_name.get(loser_name)
            if winner_id is None or loser_id is None:
                continue
            record[winner_id][mr["map_name"]][0] += 1
            record[loser_id][mr["map_name"]][1] += 1
            global_record[winner_id][0] += 1
            global_record[loser_id][1] += 1

    return decisions


def shrunk_rate(counts: list[int], global_counts: list[int], kappa: float) -> float:
    """Empirical-Bayes shrinkage of a team's map win rate toward its global
    (all-map) win rate. counts/global_counts are [wins, losses]."""
    wins, losses = counts
    n = wins + losses
    gw, gl = global_counts
    global_n = gw + gl
    global_rate = (gw / global_n) if global_n > 0 else 0.5
    denom = n + kappa
    return ((wins + kappa * global_rate) / denom) if denom > 0 else 0.5


def wins_above_expected(counts: list[int], global_counts: list[int]) -> float:
    """Wins on this map minus what you'd expect given your overall win rate
    and how many times you've played it: wins - n * global_rate. Unlike a
    rate, this naturally grows with sample size -- a team 6-2 on a map reads
    as more evidence of map strength than 3-1, even though both are 75%.
    Needs no shrinkage parameter: n=0 games gives 0 automatically (no
    evidence, no signal), so the evidence-weighting is baked into the
    feature itself rather than bolted on via kappa.
    """
    wins, losses = counts
    n = wins + losses
    gw, gl = global_counts
    global_n = gw + gl
    global_rate = (gw / global_n) if global_n > 0 else 0.5
    return wins - n * global_rate


def _eligible(decisions: list[dict], min_history_games: int) -> list[dict]:
    """Decisions where both teams have enough *global* history to estimate
    team strength at all -- shrinkage handles per-map sparsity directly, so
    this gate is now about team strength, not map-specific sample size."""
    out = []
    for d in decisions:
        own_n = sum(d["own_global"])
        opp_n = sum(d["opp_global"])
        if own_n >= min_history_games and opp_n >= min_history_games:
            out.append(d)
    return out


def _softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    s = sum(exps)
    return [e / s for e in exps]


def _build_features(
    decisions: list[dict], kappa: float, feature: str = "rate"
) -> list[tuple[int, list[float], list[float]]]:
    """(chosen_index, own_feature, opp_feature) per decision, aligned to
    d['pool'] order.

    feature="rate" (bans): shrinkage-based win rate, own negated so a positive
    coefficient reads as "more likely to act on your weak maps". Needs kappa.

    feature="wins_above_expected" (picks): evidence-weighted wins above what
    your overall record would predict, own left positive so a positive
    coefficient reads as "more likely to act on maps you've overperformed on".
    kappa is unused for this feature (ignored, kept in the signature so
    callers don't need to branch).
    """
    out = []
    for d in decisions:
        pool = d["pool"]
        chosen_idx = pool.index(d["chosen"])
        if feature == "rate":
            own = [-shrunk_rate(d["own_counts"][m], d["own_global"], kappa) for m in pool]
            opp = [shrunk_rate(d["opp_counts"][m], d["opp_global"], kappa) for m in pool]
        elif feature == "wins_above_expected":
            own = [wins_above_expected(d["own_counts"][m], d["own_global"]) for m in pool]
            opp = [wins_above_expected(d["opp_counts"][m], d["opp_global"]) for m in pool]
        else:
            raise ValueError(f"unknown feature type: {feature}")
        out.append((chosen_idx, own, opp))
    return out


def _log_likelihood(features: list[tuple[int, list[float], list[float]]], beta: tuple[float, float]) -> float:
    b0, b1 = beta
    ll = 0.0
    for chosen_idx, x1, x2 in features:
        logits = [b0 * x1[i] + b1 * x2[i] for i in range(len(x1))]
        p = _softmax(logits)
        ll += math.log(max(p[chosen_idx], 1e-12))
    return ll


def _null_log_likelihood(decisions: list[dict]) -> float:
    return sum(-math.log(len(d["pool"])) for d in decisions)


def fit_model(
    decisions: list[dict], kappa: float, feature: str = "rate", max_iters: int = 5000, grad_tol: float = 1e-7
) -> dict:
    """Fit the 2-parameter softmax model (own weakness, opponent strength) by
    gradient ascent with backtracking line search on log-likelihood. The
    objective is concave (standard multinomial logistic), so this converges
    reliably from beta=(0,0) -- but a fixed step size can still fail to
    converge if the true coefficients turn out to be large (seen with the
    pick-decision fit under the "rate" feature, which is what motivated the
    "wins_above_expected" feature -- see _build_features).
    """
    features = _build_features(decisions, kappa, feature=feature)
    beta = [0.0, 0.0]
    ll = _log_likelihood(features, beta)
    step = 1.0
    converged = False
    iterations = 0

    for iterations in range(1, max_iters + 1):
        grad = [0.0, 0.0]
        for chosen_idx, x1, x2 in features:
            logits = [beta[0] * x1[i] + beta[1] * x2[i] for i in range(len(x1))]
            p = _softmax(logits)
            ex1 = sum(p[i] * x1[i] for i in range(len(x1)))
            ex2 = sum(p[i] * x2[i] for i in range(len(x2)))
            grad[0] += x1[chosen_idx] - ex1
            grad[1] += x2[chosen_idx] - ex2
        grad[0] /= len(features)
        grad[1] /= len(features)
        grad_norm = math.hypot(*grad)
        if grad_norm < grad_tol:
            converged = True
            break

        trial = step
        moved = False
        while trial > 1e-12:
            candidate = [beta[0] + trial * grad[0], beta[1] + trial * grad[1]]
            candidate_ll = _log_likelihood(features, candidate)
            if candidate_ll >= ll:
                beta, ll = candidate, candidate_ll
                step = trial * 1.2  # grow a bit for next iteration
                moved = True
                break
            trial *= 0.5
        if not moved:
            converged = True  # step size collapsed to zero: at a local (== global, concave) optimum
            break

    return {
        "beta_own_weakness": beta[0],
        "beta_opp_strength": beta[1],
        "log_likelihood": ll,
        "converged": converged,
        "iterations": iterations,
    }


def run_likelihood_ratio_test(
    action_types: tuple[str, ...] = ("ban",),
    min_history_games: int = 3,
    kappa_grid: tuple[float, ...] = (0.5, 1, 2, 3, 5, 8, 12, 20, 40),
    test_frac: float = 0.2,
    feature: str = "rate",
) -> dict:
    """End-to-end: collect decisions, chronologically split into
    fit/kappa-select/final-test (60/20/20), pick kappa by held-out
    log-likelihood, refit on fit+select, then report the likelihood-ratio
    test (fitted vs. uniform-over-pool null) on the untouched final slice.

    p-value uses the exact chi-square(df=2) survival function, which has a
    closed form (no scipy needed): P(chi2_2 > x) = exp(-x/2).

    feature="wins_above_expected" skips the kappa search entirely (kappa is
    meaningless for that feature -- see _build_features) and just fits once
    on fit+select.
    """
    raw = collect_decisions(action_types=action_types)
    eligible = _eligible(raw, min_history_games)
    if len(eligible) < 30:
        raise ValueError(f"only {len(eligible)} eligible decisions -- too few to fit/test reliably")

    n = len(eligible)
    n_test = max(10, int(n * test_frac))
    n_select = max(10, int(n * test_frac))
    fit_only = eligible[: n - n_test - n_select]
    select = eligible[n - n_test - n_select : n - n_test]
    test = eligible[n - n_test :]

    if feature == "wins_above_expected":
        best_kappa = None
        kappa_scores = []
        final_fit = fit_model(fit_only + select, kappa=0.0, feature=feature)
    else:
        kappa_scores = []
        for kappa in kappa_grid:
            fit = fit_model(fit_only, kappa, feature=feature)
            select_features = _build_features(select, kappa, feature=feature)
            select_ll = _log_likelihood(select_features, (fit["beta_own_weakness"], fit["beta_opp_strength"]))
            kappa_scores.append((kappa, select_ll))
        best_kappa = max(kappa_scores, key=lambda t: t[1])[0]
        final_fit = fit_model(fit_only + select, best_kappa, feature=feature)

    beta = (final_fit["beta_own_weakness"], final_fit["beta_opp_strength"])

    test_features = _build_features(test, best_kappa or 0.0, feature=feature)
    fitted_ll = _log_likelihood(test_features, beta)
    null_ll = _null_log_likelihood(test)
    lr_stat = 2 * (fitted_ll - null_ll)
    p_value = math.exp(-lr_stat / 2) if lr_stat > 0 else 1.0

    return {
        "action_types": action_types,
        "feature": feature,
        "test_log_likelihood_fitted": fitted_ll,
        "test_log_likelihood_null": null_ll,
        "n_total_eligible": n,
        "n_fit": len(fit_only),
        "n_kappa_select": len(select),
        "n_test": len(test),
        "kappa_grid_scores": kappa_scores,
        "best_kappa": best_kappa,
        "beta_own_weakness": beta[0],
        "beta_opp_strength": beta[1],
        "likelihood_ratio_stat": lr_stat,
        "p_value": p_value,
        "converged": final_fit["converged"],
    }


def compare_to_naive_rule(
    action_types: tuple[str, ...] = ("ban",),
    min_history_games: int = 3,
    kappa_grid: tuple[float, ...] = (0.5, 1, 2, 3, 5, 8, 12, 20, 40),
    test_frac: float = 0.2,
) -> dict:
    """Beating random isn't the bar -- beating the one-line rule the model is
    built on top of is. Evaluates the naive rule ("act on your single
    worst/best available map") and the fitted 2-feature model on the exact
    same held-out test slice used by run_likelihood_ratio_test, same kappa.
    If the model barely beats the naive rule's accuracy, the extra
    complexity (opponent feature, continuous fit) isn't earning its keep.
    """
    raw = collect_decisions(action_types=action_types)
    eligible = _eligible(raw, min_history_games)
    n = len(eligible)
    n_test = max(10, int(n * test_frac))
    n_select = max(10, int(n * test_frac))
    fit_only = eligible[: n - n_test - n_select]
    select = eligible[n - n_test - n_select : n - n_test]
    test = eligible[n - n_test :]

    kappa_scores = []
    for kappa in kappa_grid:
        fit = fit_model(fit_only, kappa)
        select_features = _build_features(select, kappa)
        select_ll = _log_likelihood(select_features, (fit["beta_own_weakness"], fit["beta_opp_strength"]))
        kappa_scores.append((kappa, select_ll))
    best_kappa = max(kappa_scores, key=lambda t: t[1])[0]
    final_fit = fit_model(fit_only + select, best_kappa)
    beta = (final_fit["beta_own_weakness"], final_fit["beta_opp_strength"])

    test_features = _build_features(test, best_kappa)
    naive_correct = 0
    model_correct = 0
    agree = 0
    for chosen_idx, x1, x2 in test_features:
        # own-weakness feature is already -rate, so its argmax IS "act on your single worst/best map"
        naive_pred = max(range(len(x1)), key=lambda i: x1[i])
        model_logits = [beta[0] * x1[i] + beta[1] * x2[i] for i in range(len(x1))]
        model_pred = max(range(len(model_logits)), key=lambda i: model_logits[i])
        naive_correct += int(naive_pred == chosen_idx)
        model_correct += int(model_pred == chosen_idx)
        agree += int(naive_pred == model_pred)

    n_t = len(test_features)
    return {
        "n_test": n_t,
        "best_kappa": best_kappa,
        "naive_rule_accuracy": naive_correct / n_t,
        "fitted_model_accuracy": model_correct / n_t,
        "pct_decisions_where_model_and_naive_rule_agree": agree / n_t,
    }
