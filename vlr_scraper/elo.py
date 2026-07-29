"""Elo rating system: replaces global_strength_feature (base_model.py's
Model 0) as the foundation feature. Rationale (per the build instruction
this was written against): shrunk map win rate can't express "team A is
much better than team B" and is confounded with strength of schedule (a 6-0
record against weak teams looks identical to 6-0 against strong ones); Elo
fixes both. Literature check before building: on ~10k pro CS:GO matches,
Glicko-2/Elo/TrueSkill are within ~0.3pp of each other (not worth switching
systems at this sample size); per-player decomposition was the one approach
that meaningfully beat plain team Elo (~64.1% vs ~62.8-63.1%), noted as a
future direction, not built now. Tennis surface-Elo work is the direct
analogue to maps: single-surface ratings alone predict poorly, a
shrunk blend of surface-specific and overall rating works much better --
this validates the same shrinkage structure already used throughout this
codebase (veto_model.shrunk_rate, agent_proficiency's v[a,m]).

Build order, each with its own walk-forward pass or feature:
  Step 1 (build_elo_dataset, fit_elo_only)  -- plain Elo, updated per MAP
  Step 2 (residual accumulation, inside build_elo_dataset)  -- per-team-map
    shrunk residual vs. Elo's own expectation
  Step 3 (run_nested_comparison)  -- Model A (Elo alone) vs Model B (Elo +
    shrunk residual), a genuine nested comparison with a classical LR test
  Step 4 (run_step4_adjustments)  -- margin-of-victory K, dynamic K,
    mean-reversion, tested individually against Model B

CRITICAL walk-forward discipline specific to Elo: ratings update after EACH
MAP, not after the whole match/series -- a Bo3's second map is played after
the first map's result is already known in reality, so the second map's
prediction must see the first map's rating update. This differs from how
base_model.py treats team-map win-rate/agent features (updated only once
per match, after all of that match's maps) -- deliberately, because a
series' aggregate stats are conventionally treated as "not final" until the
series ends, whereas Elo ratings genuinely change map-to-map. Every
observation records its rating snapshot and expected probability BEFORE
that map's own update is applied, to avoid leaking the map's own outcome
into its own predictor.
"""

from __future__ import annotations

import math
from collections import defaultdict

from . import agent_proficiency as ap
from .stats_utils import cluster_robust_se, fit_logistic_no_intercept, log_loss, perplexity

# Pre-registered (fixed before any test was run).
# Widened after the first run: K=40 hit the original grid's edge -- per the standing
# project rule, widen and rerun rather than trust an edge value.
K_GRID = (8, 12, 16, 20, 24, 32, 40, 55, 75, 100, 130, 170)
# Widened after the first run: kappa=80 hit the original grid's edge.
RESIDUAL_KAPPA_GRID = (1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 130, 200, 320, 500, 800)
BURN_IN_DAYS = 30
TEST_FRAC = 0.2
TUNE_FRAC = 0.2
ELO_BASE = 1500.0


def _expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def build_elo_dataset(
    k: float,
    matches: list[dict] | None = None,
    mov: bool = False,
    dynamic_k: bool = False,
    dynamic_k_threshold: int = 20,
    dynamic_k_multiplier: float = 2.0,
    patch_reversion: float | None = None,
) -> list[dict]:
    """One walk-forward pass, ratings updated per map (not per match). Also
    accumulates per-(team,map) Elo residuals (Step 2) in the same pass --
    same "walk once" pattern as the rest of this codebase, since the
    residual accumulator needs the exact same rating trajectory Step 1
    produces and re-deriving it separately would risk the two drifting out
    of sync.

    Step 4 adjustments (each off by default; base_model.elo's Step 1-3 calls
    leave all of them off):
      mov -- scale the map's effective K by round-margin, ln(margin+1)
        normalized so a 1-round margin gives multiplier 1.0 (standard
        sports-Elo MOV form).
      dynamic_k -- multiply K by dynamic_k_multiplier while a team has
        played fewer than dynamic_k_threshold *rated* maps total (an
        experience count, not per-map-per-team) -- approximates Glicko's
        uncertainty-based updating without switching systems.
      patch_reversion -- if set, pull every team's rating partway back
        toward ELO_BASE (rating = rating*(1-x) + ELO_BASE*x) at each
        detected patch-version change, using the same `patch` field parsed
        off match pages elsewhere in this codebase.
    """
    if matches is None:
        matches = ap.load_matches_ordered()
    date_by_match = ap.load_date_by_match()
    patch_by_match = ap.load_patch_by_match() if patch_reversion is not None else {}

    ratings: dict[int, float] = defaultdict(lambda: ELO_BASE)
    games_played: dict[int, int] = defaultdict(int)
    residual_sum: dict[tuple[int, str], float] = defaultdict(float)
    residual_n: dict[tuple[int, str], int] = defaultdict(int)
    last_patch: str | None = None

    observations: list[dict] = []

    for m in matches:
        t1_id, t2_id = m.get("team1_id"), m.get("team2_id")
        if t1_id is None or t2_id is None:
            continue
        date = date_by_match.get(m["match_id"], "")
        team_name_to_id = {m["team1"]: t1_id, m["team2"]: t2_id}

        if patch_reversion is not None:
            current_patch = patch_by_match.get(m["match_id"])
            if current_patch is not None and last_patch is not None and current_patch != last_patch:
                for tid in list(ratings.keys()):
                    ratings[tid] = ratings[tid] * (1 - patch_reversion) + ELO_BASE * patch_reversion
            if current_patch is not None:
                last_patch = current_patch

        for mr in m["maps"]:
            if mr["winner"] is None:
                continue
            map_name = mr["map_name"]
            winner_id = team_name_to_id.get(mr["winner"])
            if winner_id is None:
                continue
            label = 1 if winner_id == t1_id else 0

            ra, rb = ratings[t1_id], ratings[t2_id]
            expected_t1 = _expected(ra, rb)
            residual_t1 = label - expected_t1  # actual - expected, this map, BEFORE the update

            rounds_t1 = mr.get("team1_score") if mr["team1"] == m["team1"] else mr.get("team2_score")
            rounds_t2 = mr.get("team2_score") if mr["team1"] == m["team1"] else mr.get("team1_score")

            observations.append(
                {
                    "match_id": m["match_id"],
                    "map_name": map_name,
                    "date": date,
                    "t1_id": t1_id,
                    "t2_id": t2_id,
                    "label": label,
                    "elo_diff": ra - rb,
                    "expected_t1": expected_t1,
                    "t1_residual_sum": residual_sum[(t1_id, map_name)],
                    "t1_residual_n": residual_n[(t1_id, map_name)],
                    "t2_residual_sum": residual_sum[(t2_id, map_name)],
                    "t2_residual_n": residual_n[(t2_id, map_name)],
                    "rounds_t1": rounds_t1,
                    "rounds_t2": rounds_t2,
                }
            )

            k_eff = k
            if mov and rounds_t1 is not None and rounds_t2 is not None:
                margin = abs(rounds_t1 - rounds_t2)
                k_eff *= math.log(margin + 1) / math.log(2)  # margin=1 -> multiplier 1.0
            if dynamic_k:
                if games_played[t1_id] < dynamic_k_threshold or games_played[t2_id] < dynamic_k_threshold:
                    k_eff *= dynamic_k_multiplier

            # Update ratings AFTER recording this map's pre-update snapshot -- per-map, not
            # per-match, so the next map in the same series (if any) sees this update.
            ratings[t1_id] = ra + k_eff * (label - expected_t1)
            ratings[t2_id] = rb + k_eff * ((1 - label) - (1 - expected_t1))
            games_played[t1_id] += 1
            games_played[t2_id] += 1
            residual_sum[(t1_id, map_name)] += residual_t1
            residual_n[(t1_id, map_name)] += 1
            residual_sum[(t2_id, map_name)] += -residual_t1
            residual_n[(t2_id, map_name)] += 1

    return observations


def apply_burn_in(observations: list[dict], burn_in_days: int = BURN_IN_DAYS) -> list[dict]:
    dated = [o["date"] for o in observations if o["date"]]
    if not dated:
        return observations
    start = min(dated)
    from datetime import date as _date, timedelta

    y, mo, da = start[:10].split("-")
    cutoff = (_date(int(y), int(mo), int(da)) + timedelta(days=burn_in_days)).isoformat()
    return [o for o in observations if o["date"] and o["date"] >= cutoff]


def _chronological_split(observations: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    n = len(observations)
    n_test = max(20, int(n * TEST_FRAC))
    n_tune = max(20, int(n * TUNE_FRAC))
    return observations[: n - n_test - n_tune], observations[n - n_test - n_tune : n - n_test], observations[n - n_test :]


def elo_feature(o: dict) -> float:
    return o["elo_diff"]


def shrunk_residual_feature(o: dict, kappa: float) -> float:
    t1 = o["t1_residual_sum"] / (o["t1_residual_n"] + kappa) if (o["t1_residual_n"] + kappa) > 0 else 0.0
    t2 = o["t2_residual_sum"] / (o["t2_residual_n"] + kappa) if (o["t2_residual_n"] + kappa) > 0 else 0.0
    return t1 - t2


def _quantiles(vals: list[float]) -> dict:
    v = sorted(vals)
    n = len(v)

    def q(p: float) -> float:
        idx = p * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        frac = idx - lo
        return v[lo] * (1 - frac) + v[hi] * frac

    return {"min": v[0], "q25": q(0.25), "median": q(0.5), "q75": q(0.75), "max": v[-1]}


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def fit_elo_only(verbose: bool = True) -> dict:
    """Step 1: tune K on the tune slice by held-out log loss, fit the
    single-feature (elo_diff) no-intercept logistic model on fit+tune,
    report test-slice log loss/perplexity and the predicted-probability
    spread (min/quartiles/max) -- the "can it produce confident predictions"
    check the instruction specifically asked for.
    """
    k_scores = []
    datasets_by_k: dict[float, list[dict]] = {}
    for k in K_GRID:
        obs = apply_burn_in(build_elo_dataset(k))
        datasets_by_k[k] = obs
        fit_obs, tune_obs, test_obs = _chronological_split(obs)
        X = [[elo_feature(o)] for o in fit_obs]
        y = [o["label"] for o in fit_obs]
        fit = fit_logistic_no_intercept(X, y)
        X_tune = [[elo_feature(o)] for o in tune_obs]
        y_tune = [o["label"] for o in tune_obs]
        k_scores.append((k, log_loss(X_tune, y_tune, fit["beta"])))
        if verbose:
            print(f"  K={k}: tune log loss={k_scores[-1][1]:.5f}", file=__import__("sys").stderr)
    best_k = min(k_scores, key=lambda t: t[1])[0]
    grid_min, grid_max = min(K_GRID), max(K_GRID)

    obs = datasets_by_k[best_k]
    fit_obs, tune_obs, test_obs = _chronological_split(obs)
    fit_plus_tune = fit_obs + tune_obs

    X = [[elo_feature(o)] for o in fit_plus_tune]
    y = [o["label"] for o in fit_plus_tune]
    final_fit = fit_logistic_no_intercept(X, y)
    se = cluster_robust_se(final_fit["info_matrix"], final_fit["scores"], [o["match_id"] for o in fit_plus_tune])

    X_test = [[elo_feature(o)] for o in test_obs]
    y_test = [o["label"] for o in test_obs]
    test_ll = log_loss(X_test, y_test, final_fit["beta"])
    beta = final_fit["beta"][0]
    predicted_probs = [sigmoid(beta * x[0]) for x in X_test]

    return {
        "k_grid_scores": k_scores,
        "best_k": best_k,
        "k_grid_range": [grid_min, grid_max],
        "k_is_interior": grid_min < best_k < grid_max,
        "n_total_after_burn_in": len(obs),
        "n_fit": len(fit_obs),
        "n_tune": len(tune_obs),
        "n_test": len(test_obs),
        "beta_elo_diff": beta,
        "beta_elo_diff_cluster_se": se[0],
        "implied_elo_scale_check": {"fitted_beta": beta, "classical_elo_beta": math.log(10) / 400},
        "converged": final_fit["converged"],
        "test_log_loss_nats": test_ll,
        "test_perplexity": perplexity(test_ll),
        "predicted_probability_spread": _quantiles(predicted_probs),
    }


def run_nested_comparison(best_k: float, verbose: bool = True) -> dict:
    """Step 3: Model A (elo_diff alone) vs Model B (elo_diff + shrunk
    residual diff) -- B strictly contains A's feature plus one more, so this
    is a genuine nested comparison (unlike base_model's earlier Model 0 vs
    Model 1, which compared two different single-feature specifications).
    Reports the held-out log-loss difference AND a classical nested
    likelihood-ratio test (both models fit by MLE on the same fit+tune
    sample, df=1 chi-square on twice the log-likelihood gain from adding the
    residual coefficient) -- the standard test for this exact situation,
    distinct from veto_model.py's held-out-vs-uniform-null LR test.
    """
    obs = apply_burn_in(build_elo_dataset(best_k))
    fit_obs, tune_obs, test_obs = _chronological_split(obs)

    # --- tune residual kappa on tune slice, Model B ---
    kappa_scores = []
    for kappa in RESIDUAL_KAPPA_GRID:
        X = [[elo_feature(o), shrunk_residual_feature(o, kappa)] for o in fit_obs]
        y = [o["label"] for o in fit_obs]
        fit = fit_logistic_no_intercept(X, y)
        X_tune = [[elo_feature(o), shrunk_residual_feature(o, kappa)] for o in tune_obs]
        y_tune = [o["label"] for o in tune_obs]
        kappa_scores.append((kappa, log_loss(X_tune, y_tune, fit["beta"])))
        if verbose:
            print(f"  residual kappa={kappa}: tune log loss={kappa_scores[-1][1]:.5f}", file=__import__("sys").stderr)
    best_kappa = min(kappa_scores, key=lambda t: t[1])[0]
    grid_min, grid_max = min(RESIDUAL_KAPPA_GRID), max(RESIDUAL_KAPPA_GRID)

    fit_plus_tune = fit_obs + tune_obs

    # --- Model A: elo alone, fit on fit+tune ---
    Xa = [[elo_feature(o)] for o in fit_plus_tune]
    ya = [o["label"] for o in fit_plus_tune]
    fit_a = fit_logistic_no_intercept(Xa, ya)

    # --- Model B: elo + shrunk residual, fit on fit+tune ---
    Xb = [[elo_feature(o), shrunk_residual_feature(o, best_kappa)] for o in fit_plus_tune]
    yb = [o["label"] for o in fit_plus_tune]
    fit_b = fit_logistic_no_intercept(Xb, yb)
    se_b = cluster_robust_se(fit_b["info_matrix"], fit_b["scores"], [o["match_id"] for o in fit_plus_tune])

    # --- classical nested LR test (training log-likelihoods, df=1) ---
    lr_stat = 2 * (fit_b["log_likelihood"] - fit_a["log_likelihood"])
    lr_stat = max(lr_stat, 0.0)  # numerical guard: B >= A in-sample by construction
    p_value = math.erfc(math.sqrt(lr_stat / 2)) if lr_stat > 0 else 1.0

    # --- held-out (test-slice) comparison ---
    Xa_test = [[elo_feature(o)] for o in test_obs]
    y_test = [o["label"] for o in test_obs]
    test_ll_a = log_loss(Xa_test, y_test, fit_a["beta"])

    Xb_test = [[elo_feature(o), shrunk_residual_feature(o, best_kappa)] for o in test_obs]
    test_ll_b = log_loss(Xb_test, y_test, fit_b["beta"])

    return {
        "best_k_used": best_k,
        "residual_kappa_grid_scores": kappa_scores,
        "best_residual_kappa": best_kappa,
        "residual_kappa_grid_range": [grid_min, grid_max],
        "residual_kappa_is_interior": grid_min < best_kappa < grid_max,
        "n_fit": len(fit_obs), "n_tune": len(tune_obs), "n_test": len(test_obs),
        "model_a_elo_only": {
            "beta_elo": fit_a["beta"][0],
            "converged": fit_a["converged"],
            "test_log_loss_nats": test_ll_a,
            "test_perplexity": perplexity(test_ll_a),
        },
        "model_b_elo_plus_residual": {
            "beta_elo": fit_b["beta"][0],
            "beta_elo_cluster_se": se_b[0],
            "beta_residual": fit_b["beta"][1],
            "beta_residual_cluster_se": se_b[1],
            "beta_residual_z": fit_b["beta"][1] / se_b[1] if se_b[1] > 0 else None,
            "converged": fit_b["converged"],
            "test_log_loss_nats": test_ll_b,
            "test_perplexity": perplexity(test_ll_b),
        },
        "held_out_improvement_nats": test_ll_a - test_ll_b,
        "nested_likelihood_ratio_test": {
            "description": "Classical nested LR test: both models fit by MLE on fit+tune, df=1 chi-square on 2*(llB-llA).",
            "ll_a_training": fit_a["log_likelihood"],
            "ll_b_training": fit_b["log_likelihood"],
            "lr_statistic": lr_stat,
            "p_value": p_value,
        },
    }


def _fit_and_eval_model_b(obs: list[dict], best_kappa: float) -> dict:
    fit_obs, tune_obs, test_obs = _chronological_split(obs)
    fit_plus_tune = fit_obs + tune_obs
    X = [[elo_feature(o), shrunk_residual_feature(o, best_kappa)] for o in fit_plus_tune]
    y = [o["label"] for o in fit_plus_tune]
    fit = fit_logistic_no_intercept(X, y)
    X_test = [[elo_feature(o), shrunk_residual_feature(o, best_kappa)] for o in test_obs]
    y_test = [o["label"] for o in test_obs]
    test_ll = log_loss(X_test, y_test, fit["beta"])
    return {
        "n_fit": len(fit_obs), "n_tune": len(tune_obs), "n_test": len(test_obs),
        "beta_elo": fit["beta"][0], "beta_residual": fit["beta"][1],
        "converged": fit["converged"],
        "test_log_loss_nats": test_ll,
        "test_perplexity": perplexity(test_ll),
    }


def run_step4_adjustments(best_k: float = 40, best_kappa: float = 130, verbose: bool = True) -> dict:
    """Step 4: each adjustment tested individually against the Step 3
    baseline (Model B: elo + shrunk residual, unadjusted K). K and the
    residual kappa are held fixed at their Step 1/3 values throughout --
    the point is to isolate each adjustment's own effect, not to re-tune
    everything around it.
    """
    baseline_obs = apply_burn_in(build_elo_dataset(best_k))
    baseline_result = _fit_and_eval_model_b(baseline_obs, best_kappa)
    if verbose:
        print(f"  baseline (Step 3, unadjusted): test log loss={baseline_result['test_log_loss_nats']:.5f}", file=__import__("sys").stderr)

    mov_obs = apply_burn_in(build_elo_dataset(best_k, mov=True))
    mov_result = _fit_and_eval_model_b(mov_obs, best_kappa)
    if verbose:
        print(f"  (a) margin-of-victory: test log loss={mov_result['test_log_loss_nats']:.5f}", file=__import__("sys").stderr)

    dynk_obs = apply_burn_in(build_elo_dataset(best_k, dynamic_k=True))
    dynk_result = _fit_and_eval_model_b(dynk_obs, best_kappa)
    if verbose:
        print(f"  (b) dynamic K: test log loss={dynk_result['test_log_loss_nats']:.5f}", file=__import__("sys").stderr)

    reversion_obs = apply_burn_in(build_elo_dataset(best_k, patch_reversion=0.25))
    reversion_result = _fit_and_eval_model_b(reversion_obs, best_kappa)
    if verbose:
        print(f"  (c) patch-boundary reversion: test log loss={reversion_result['test_log_loss_nats']:.5f}", file=__import__("sys").stderr)

    return {
        "baseline_step3": baseline_result,
        "a_margin_of_victory": mov_result,
        "a_improvement_nats": baseline_result["test_log_loss_nats"] - mov_result["test_log_loss_nats"],
        "b_dynamic_k": dynk_result,
        "b_improvement_nats": baseline_result["test_log_loss_nats"] - dynk_result["test_log_loss_nats"],
        "c_patch_reversion_025": reversion_result,
        "c_improvement_nats": baseline_result["test_log_loss_nats"] - reversion_result["test_log_loss_nats"],
    }
