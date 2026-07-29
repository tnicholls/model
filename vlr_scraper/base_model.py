"""Model 1: P(team wins | map) -- the base model the whole pricing chain
depends on (per the build spec's priority note: "Even if the agent features
add nothing, this task produces a fitted P(win | map) estimator, which is
currently the largest missing piece of the overall pipeline"). Build this
first, confirm it works, only then layer on the agent-strength machinery.

Baseline feature: differenced shrunk map win-rate (veto_model.shrunk_rate).
Extended feature: + differenced, within-team-standardized agent_strength
(the factorised proficiency estimate from agent_proficiency.py).

Single walk-forward pass (build_observations) records RAW ingredients for
both features *as of the moment before each match's results are known* --
kappa/shrinkage tuning is then a fast in-memory pass over those raw
ingredients, same "walk once, tune cheaply" pattern as veto_model.py.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from . import agent_proficiency as ap
from .agent_proficiency import (
    MAP_ROTATION_PATH,
    PATCH_DATES_PATH,
    STALE_MIN_TEAM_MAP_GAMES,
    STALE_ROTATION_WINDOW_DAYS,
    _days_between,
    most_recent_pool_entry_at,
    most_recent_patch_at,
)
from .stats_utils import cluster_robust_se, fit_logistic_no_intercept, log_loss, perplexity
from .veto import TEAM_TAGS_PATH, VetoStep, _resolve_veto_team
from .veto_model import shrunk_rate

# Pre-registered hyperparameter grids (fixed before any test was run).
# Widened after the first run: kappa_map hit the original grid's edge (40) -- per spec,
# "if a tuned kappa lands at the edge of its grid, widen the grid and rerun."
KAPPA_MAP_GRID = (0.3, 0.5, 1, 2, 3, 5, 8, 12, 20, 40, 80, 160, 320, 640, 1280)  # 0.3: the ban model's found optimum
KAPPA_PLAYER_GRID = ap.KAPPA_PLAYER_GRID
KAPPA_AGENT_MAP_GRID = ap.KAPPA_AGENT_MAP_GRID
TEST_FRAC = 0.2
TUNE_FRAC = 0.2


# --------------------------------------------------------------------------
# Walk-forward pass: build raw observations
# --------------------------------------------------------------------------


def _argmax_sorted(counts: dict) -> str | None:
    """Highest-count key, ties broken alphabetically for determinism (see
    veto.test_ban_rules' comment on why: Python's per-process string-hash
    randomization otherwise makes tie-breaks non-reproducible)."""
    if not counts:
        return None
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _player_typical_agent(player_id: int, map_name: str, player_agent_map_count: dict, player_agent_count: dict) -> str | None:
    per_map = player_agent_map_count.get(player_id, {}).get(map_name)
    best = _argmax_sorted(per_map) if per_map else None
    if best:
        return best
    return _argmax_sorted(player_agent_count.get(player_id, {}))


def _agent_terms_for_team(
    roster: list[int],
    team_id: int,
    target_map: str,
    team_map_wl: dict,
    player_agent_map_count: dict,
    player_agent_count: dict,
    player_agent_rating: dict,
    agent_map_wl: dict,
    agent_global_wl: dict,
    agent_rating: dict,
) -> dict[str, list]:
    """For team_id's roster, raw (pi_z, pi_n, v_diff, v_n) terms per player,
    for target_map and every other map the team has history on (needed to
    standardize agent_strength across maps within the team -- Step 4 of the
    spec). None in place of a term means that player has no usable history
    (no typical agent yet, or that agent has <2 rated appearances anywhere)."""
    # sorted(), not raw set iteration: summing floating-point terms in a hash-randomized
    # order (set iteration order varies per process for strings) produces last-bit-different
    # results across runs -- same root cause as the tie-break non-determinism documented in
    # veto.test_ban_rules, here showing up as float non-associativity instead.
    candidate_maps = sorted(set(team_map_wl.get(team_id, {}).keys()) | {target_map})
    out: dict[str, list] = {}
    for mp in candidate_maps:
        terms = []
        for player_id in roster:
            agent = _player_typical_agent(player_id, mp, player_agent_map_count, player_agent_count)
            if agent is None:
                terms.append(None)
                continue
            pa = player_agent_rating.get((player_id, agent))
            a_stat = agent_rating.get(agent)
            amap = agent_map_wl.get((agent, mp))
            if pa is None or a_stat is None or amap is None:
                terms.append(None)
                continue
            pi_n = pa[3]
            a_n = a_stat[3]
            if pi_n <= 0 or a_n <= 1:
                terms.append(None)
                continue
            pi_mean = pa[1] / pa[0]  # sum_wr / sum_w
            a_mean = a_stat[1] / a_stat[0]
            a_var = a_stat[2] / a_stat[0] - a_mean**2
            a_sd = math.sqrt(a_var) if a_var > 0 else 1.0
            pi_z = (pi_mean - a_mean) / a_sd

            amap_w, amap_l = amap
            amap_n = amap_w + amap_l
            aglob_w, aglob_l = agent_global_wl.get(agent, [0, 0])
            aglob_n = aglob_w + aglob_l
            if amap_n == 0 or aglob_n == 0:
                terms.append(None)
                continue
            v_diff = (amap_w / amap_n) - (aglob_w / aglob_n)
            terms.append((pi_z, pi_n, v_diff, amap_n))
        out[mp] = terms
    return out


def _load_match_rosters(tag_cache: dict) -> dict[tuple[int, str, int], list[tuple[int, str, float | None, int | None]]]:
    """(match_id, map_name, team_id) -> [(player_id, agent, rating, rounds_played), ...]
    from player_map_stats, team_tag resolved via the authoritative tag cache
    (same resolution veto.py uses for veto-note labels)."""
    out: dict[tuple[int, str, int], list] = defaultdict(list)
    for line in ap.PLAYER_MAP_STATS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        team_id = _resolve_veto_team(r["team_tag"], r["team1_id"], r["team2_id"], tag_cache)
        if team_id is None:
            continue
        out[(r["match_id"], r["map_name"], team_id)].append((r["player_id"], r["agent"], r["rating"], r["rounds_played"]))
    return out


def _get_since_patch(since_patch: dict, team_id: int, map_name: str, patch: str | None) -> int:
    if patch is None:
        return 0
    rec = since_patch.get((team_id, map_name))
    if rec is None or rec["patch"] != patch:
        return 0
    return rec["n"]


def _update_since_patch(since_patch: dict, team_id: int, map_name: str, patch: str | None) -> None:
    if patch is None:
        return
    rec = since_patch.setdefault((team_id, map_name), {"patch": patch, "n": 0})
    if rec["patch"] != patch:
        rec["patch"] = patch
        rec["n"] = 0
    rec["n"] += 1


def _match_team_roster(rosters: dict, match_id: int, team_id: int) -> list[int]:
    """All player_ids that appeared for team_id anywhere in this match (union
    across maps) -- used as "this match's roster" for veto decisions, where
    the map in question may never actually get played (it could be banned),
    so there's no per-map roster to key off of the way completed-map
    observations can."""
    seen: dict[int, None] = {}
    for (mid, _mp, tid), rows in rosters.items():
        if mid == match_id and tid == team_id:
            for player_id, _a, _r, _rd in rows:
                seen[player_id] = None
    return list(seen)


def build_observations() -> list[dict]:
    return build_walkforward()["map_observations"]


def build_walkforward() -> dict:
    """One walk-forward pass over every match (canonical chronological
    order), producing two aligned datasets from the same state:

      * map_observations -- one row per completed map result, for Model 1
        (P(win | map), this module's own fit/evaluate pipeline below).
      * veto_decisions -- one row per ban/pick decision, extending
        veto_model.collect_decisions with agent_strength raw terms for every
        map still in the pool at decision time (own-team and opponent), for
        Test 2 (does agent_strength help the existing ban model). Uses the
        same own/opp win-loss bookkeeping veto_model.collect_decisions does,
        duplicated here (rather than imported) because it needs to interleave
        with this module's own state update, which must run exactly once per
        match for both datasets to see a consistent point-in-time snapshot.

    State is updated with a match's actual results only after everything
    that match needs to report on has been recorded -- no lookahead, same
    pattern as veto.collect_veto_dataset / veto_model.collect_decisions.
    """
    tag_cache = json.loads(TEAM_TAGS_PATH.read_text(encoding="utf-8"))
    matches = ap.load_matches_ordered()
    date_by_match = ap.load_date_by_match()
    patch_by_match = ap.load_patch_by_match()
    patch_dates = json.loads(PATCH_DATES_PATH.read_text(encoding="utf-8")) if PATCH_DATES_PATH.exists() else {}
    map_rotation = json.loads(MAP_ROTATION_PATH.read_text(encoding="utf-8")) if MAP_ROTATION_PATH.exists() else {}
    rosters = _load_match_rosters(tag_cache)

    team_map_wl: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    team_global_wl: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    agent_map_wl: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    agent_global_wl: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    player_agent_map_count: dict[int, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    player_agent_count: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # [sum_w, sum_w*rating, sum_w*rating^2, n_games] -- weighted by rounds_played (spec: a
    # 13-2 stomp's rating is less informative than a 13-11 game's, so weight by rounds)
    player_agent_rating: dict[tuple[int, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    agent_rating: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    since_patch: dict[tuple[int, str], dict] = {}

    map_observations: list[dict] = []
    veto_decisions: list[dict] = []

    for m in matches:
        match_id = m["match_id"]
        t1_id, t2_id = m.get("team1_id"), m.get("team2_id")
        if t1_id is None or t2_id is None:
            continue
        date = date_by_match.get(match_id, "")
        patch = patch_by_match.get(match_id) or (most_recent_patch_at(date, patch_dates) if date else None)
        team_name_to_id = {m["team1"]: t1_id, m["team2"]: t2_id}
        completed_maps = [mr for mr in m["maps"] if mr["winner"] is not None]

        for mr in completed_maps:
            map_name = mr["map_name"]
            label = 1 if mr["winner"] == m["team1"] else 0

            # n_since_patch kept as a diagnostic field only (see STALE_MIN_TEAM_MAP_GAMES'
            # comment on why raw team-map sample count, not patch-recency, drives `stale`).
            n_since_patch_t1 = _get_since_patch(since_patch, t1_id, map_name, patch)
            n_since_patch_t2 = _get_since_patch(since_patch, t2_id, map_name, patch)
            n_team_map_t1 = sum(team_map_wl[t1_id][map_name])
            n_team_map_t2 = sum(team_map_wl[t2_id][map_name])
            days_since_entry = None
            if date:
                entry_date = most_recent_pool_entry_at(map_name, date, map_rotation)
                if entry_date:
                    days_since_entry = _days_between(entry_date, date)
            stale = (
                n_team_map_t1 < STALE_MIN_TEAM_MAP_GAMES
                or n_team_map_t2 < STALE_MIN_TEAM_MAP_GAMES
                or (days_since_entry is not None and days_since_entry <= STALE_ROTATION_WINDOW_DAYS)
            )

            roster_t1 = [p for p, _a, _r, _rd in rosters.get((match_id, map_name, t1_id), [])]
            roster_t2 = [p for p, _a, _r, _rd in rosters.get((match_id, map_name, t2_id), [])]
            t1_terms = _agent_terms_for_team(
                roster_t1, t1_id, map_name, team_map_wl, player_agent_map_count, player_agent_count,
                player_agent_rating, agent_map_wl, agent_global_wl, agent_rating,
            )
            t2_terms = _agent_terms_for_team(
                roster_t2, t2_id, map_name, team_map_wl, player_agent_map_count, player_agent_count,
                player_agent_rating, agent_map_wl, agent_global_wl, agent_rating,
            )

            map_observations.append(
                {
                    "match_id": match_id,
                    "map_name": map_name,
                    "date": date,
                    "patch": patch,
                    "label": label,
                    "stale": stale,
                    "n_team_map_t1": n_team_map_t1,
                    "n_team_map_t2": n_team_map_t2,
                    "n_since_patch_t1": n_since_patch_t1,
                    "n_since_patch_t2": n_since_patch_t2,
                    "days_since_pool_entry": days_since_entry,
                    "t1_map_counts": list(team_map_wl[t1_id][map_name]),
                    "t1_global_counts": list(team_global_wl[t1_id]),
                    "t2_map_counts": list(team_map_wl[t2_id][map_name]),
                    "t2_global_counts": list(team_global_wl[t2_id]),
                    "t1_agent_terms": t1_terms,
                    "t2_agent_terms": t2_terms,
                }
            )

        # --- veto ban/pick decisions (Test 2 material), same pre-match state ---
        veto_steps = [VetoStep(**s) for s in m.get("veto", [])]
        if veto_steps:
            tag_cache_local = tag_cache  # already loaded above
            available = {s.map for s in veto_steps}
            position: dict[str, int] = defaultdict(int)
            roster_t1_match = _match_team_roster(rosters, match_id, t1_id)
            roster_t2_match = _match_team_roster(rosters, match_id, t2_id)

            for step in veto_steps:
                if step.action not in ("ban", "pick"):
                    continue
                position[step.action] += 1
                if step.action in ("ban", "pick") and len(available) >= 2:
                    team_id = _resolve_veto_team(step.team, t1_id, t2_id, tag_cache_local)
                    if team_id is not None:
                        opp_id = t2_id if team_id == t1_id else t1_id
                        own_roster = roster_t1_match if team_id == t1_id else roster_t2_match
                        opp_roster = roster_t2_match if team_id == t1_id else roster_t1_match
                        pool = sorted(available)
                        own_terms_by_map = {
                            mp: _agent_terms_for_team(
                                own_roster, team_id, mp, team_map_wl, player_agent_map_count, player_agent_count,
                                player_agent_rating, agent_map_wl, agent_global_wl, agent_rating,
                            )[mp]
                            for mp in pool
                        }
                        opp_terms_by_map = {
                            mp: _agent_terms_for_team(
                                opp_roster, opp_id, mp, team_map_wl, player_agent_map_count, player_agent_count,
                                player_agent_rating, agent_map_wl, agent_global_wl, agent_rating,
                            )[mp]
                            for mp in pool
                        }
                        chosen_n_team_map_own = sum(team_map_wl[team_id][step.map])
                        chosen_n_team_map_opp = sum(team_map_wl[opp_id][step.map])
                        chosen_days_since_entry = None
                        if date:
                            entry_date = most_recent_pool_entry_at(step.map, date, map_rotation)
                            if entry_date:
                                chosen_days_since_entry = _days_between(entry_date, date)
                        chosen_stale = (
                            chosen_n_team_map_own < STALE_MIN_TEAM_MAP_GAMES
                            or chosen_n_team_map_opp < STALE_MIN_TEAM_MAP_GAMES
                            or (chosen_days_since_entry is not None and chosen_days_since_entry <= STALE_ROTATION_WINDOW_DAYS)
                        )
                        veto_decisions.append(
                            {
                                "match_id": match_id,
                                "date": date,
                                "patch": patch,
                                "action": step.action,
                                "position": position[step.action],
                                "pool": pool,
                                "chosen": step.map,
                                "stale": chosen_stale,
                                "own_counts": {mp: list(team_map_wl[team_id][mp]) for mp in pool},
                                "opp_counts": {mp: list(team_map_wl[opp_id][mp]) for mp in pool},
                                "own_global": list(team_global_wl[team_id]),
                                "opp_global": list(team_global_wl[opp_id]),
                                "own_agent_terms_by_map": own_terms_by_map,
                                "opp_agent_terms_by_map": opp_terms_by_map,
                            }
                        )
                available.discard(step.map)

        # Update state with this match's actual results only after all of its maps are recorded.
        for mr in completed_maps:
            map_name = mr["map_name"]
            winner_id = team_name_to_id.get(mr["winner"])
            loser_name = m["team1"] if mr["winner"] == m["team2"] else m["team2"]
            loser_id = team_name_to_id.get(loser_name)
            if winner_id is None or loser_id is None:
                continue
            team_map_wl[winner_id][map_name][0] += 1
            team_map_wl[loser_id][map_name][1] += 1
            team_global_wl[winner_id][0] += 1
            team_global_wl[loser_id][1] += 1
            _update_since_patch(since_patch, winner_id, map_name, patch)
            _update_since_patch(since_patch, loser_id, map_name, patch)

            for team_id, won in ((winner_id, True), (loser_id, False)):
                for player_id, agent, rating, rounds_played in rosters.get((match_id, map_name, team_id), []):
                    if agent is None:
                        continue
                    agent_map_wl[(agent, map_name)][0 if won else 1] += 1
                    agent_global_wl[agent][0 if won else 1] += 1
                    player_agent_map_count[player_id][map_name][agent] += 1
                    player_agent_count[player_id][agent] += 1
                    if rating is not None:
                        w = rounds_played if rounds_played else 1
                        s = player_agent_rating[(player_id, agent)]
                        s[0] += w
                        s[1] += w * rating
                        s[2] += w * rating * rating
                        s[3] += 1
                        g = agent_rating[agent]
                        g[0] += w
                        g[1] += w * rating
                        g[2] += w * rating * rating
                        g[3] += 1

    return {"map_observations": map_observations, "veto_decisions": veto_decisions}


# --------------------------------------------------------------------------
# Kappa application (post-hoc, cheap -- no re-walking needed)
# --------------------------------------------------------------------------


def _agent_strength_value(terms_for_map: list, kappa_player: float, kappa_agent_map: float) -> float:
    total = 0.0
    for t in terms_for_map:
        if t is None:
            continue
        pi_z, pi_n, v_diff, v_n = t
        pi_hat = pi_z * pi_n / (pi_n + kappa_player)
        v_hat = v_diff * v_n / (v_n + kappa_agent_map)
        total += pi_hat * v_hat
    return total


def _standardized_agent_strength(agent_terms_by_map: dict, target_map: str, kappa_player: float, kappa_agent_map: float) -> float:
    values = {mp: _agent_strength_value(terms, kappa_player, kappa_agent_map) for mp, terms in agent_terms_by_map.items()}
    vals = list(values.values())
    if len(vals) < 2:
        return values[target_map]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    sd = math.sqrt(var) if var > 0 else 1.0
    return (values[target_map] - mean) / sd


def baseline_feature(obs: dict, kappa_map: float) -> float:
    r1 = shrunk_rate(obs["t1_map_counts"], obs["t1_global_counts"], kappa_map)
    r2 = shrunk_rate(obs["t2_map_counts"], obs["t2_global_counts"], kappa_map)
    return r1 - r2


def global_strength_feature(obs: dict) -> float:
    """Diagnostic-only benchmark: team global (map-agnostic) win rate diff --
    equivalent to baseline_feature at kappa_map -> infinity. Calibrates what
    the baseline's held-out log loss actually means: if this does about as
    well as the map-aware baseline, most of the baseline's predictive power
    is just "which team is better overall", not map identity, and 0.675
    nats/perplexity~1.96 reflects thin per-map data more than a weak model."""
    gw1, gl1 = obs["t1_global_counts"]
    gw2, gl2 = obs["t2_global_counts"]
    r1 = gw1 / (gw1 + gl1) if (gw1 + gl1) > 0 else 0.5
    r2 = gw2 / (gw2 + gl2) if (gw2 + gl2) > 0 else 0.5
    return r1 - r2


def agent_feature(obs: dict, kappa_player: float, kappa_agent_map: float) -> float:
    s1 = _standardized_agent_strength(obs["t1_agent_terms"], obs["map_name"], kappa_player, kappa_agent_map)
    s2 = _standardized_agent_strength(obs["t2_agent_terms"], obs["map_name"], kappa_player, kappa_agent_map)
    return s1 - s2


# --------------------------------------------------------------------------
# End-to-end: split, tune, fit, evaluate
# --------------------------------------------------------------------------


def _chronological_split(observations: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    n = len(observations)
    n_test = max(20, int(n * TEST_FRAC))
    n_tune = max(20, int(n * TUNE_FRAC))
    fit = observations[: n - n_test - n_tune]
    tune = observations[n - n_test - n_tune : n - n_test]
    test = observations[n - n_test :]
    return fit, tune, test


def _subset_log_loss(model_key: str, obs_list: list[dict], beta: list[float], kappa_map: float, kp: float, ka: float) -> tuple[float | None, int]:
    if not obs_list:
        return None, 0
    X, y = [], []
    for o in obs_list:
        if model_key == "global":
            feats = [global_strength_feature(o)]
        else:
            feats = [baseline_feature(o, kappa_map)]
            if model_key == "extended":
                feats.append(agent_feature(o, kp, ka))
        X.append(feats)
        y.append(o["label"])
    return log_loss(X, y, beta), len(y)


def run(verbose: bool = True, observations: list[dict] | None = None) -> dict:
    observations = observations if observations is not None else build_observations()
    fit_obs, tune_obs, test_obs = _chronological_split(observations)
    if verbose:
        print(f"  observations: {len(observations)} (fit={len(fit_obs)}, tune={len(tune_obs)}, test={len(test_obs)})", file=__import__("sys").stderr)

    # --- tune kappa_map on baseline model, evaluated on tune slice ---
    kappa_map_scores = []
    for km in KAPPA_MAP_GRID:
        X = [[baseline_feature(o, km)] for o in fit_obs]
        y = [o["label"] for o in fit_obs]
        fit = fit_logistic_no_intercept(X, y)
        X_tune = [[baseline_feature(o, km)] for o in tune_obs]
        y_tune = [o["label"] for o in tune_obs]
        kappa_map_scores.append((km, log_loss(X_tune, y_tune, fit["beta"])))
    best_kappa_map = min(kappa_map_scores, key=lambda t: t[1])[0]

    # --- tune (kappa_player, kappa_agent_map) jointly on extended model, evaluated on tune slice ---
    agent_kappa_scores = []
    for kp in KAPPA_PLAYER_GRID:
        for ka in KAPPA_AGENT_MAP_GRID:
            X = [[baseline_feature(o, best_kappa_map), agent_feature(o, kp, ka)] for o in fit_obs]
            y = [o["label"] for o in fit_obs]
            fit = fit_logistic_no_intercept(X, y)
            X_tune = [[baseline_feature(o, best_kappa_map), agent_feature(o, kp, ka)] for o in tune_obs]
            y_tune = [o["label"] for o in tune_obs]
            agent_kappa_scores.append((kp, ka, log_loss(X_tune, y_tune, fit["beta"])))
    best_kp, best_ka, _ = min(agent_kappa_scores, key=lambda t: t[2])

    fit_plus_tune = fit_obs + tune_obs

    # --- Model 0 (diagnostic): global team strength only, no map identity at all ---
    Xg = [[global_strength_feature(o)] for o in fit_plus_tune]
    yg = [o["label"] for o in fit_plus_tune]
    global_fit = fit_logistic_no_intercept(Xg, yg)

    # --- final baseline fit ---
    Xb = [[baseline_feature(o, best_kappa_map)] for o in fit_plus_tune]
    yb = [o["label"] for o in fit_plus_tune]
    baseline_fit = fit_logistic_no_intercept(Xb, yb)
    baseline_cluster_se = cluster_robust_se(baseline_fit["info_matrix"], baseline_fit["scores"], [o["match_id"] for o in fit_plus_tune])

    # --- final extended fit ---
    Xe = [[baseline_feature(o, best_kappa_map), agent_feature(o, best_kp, best_ka)] for o in fit_plus_tune]
    ye = [o["label"] for o in fit_plus_tune]
    extended_fit = fit_logistic_no_intercept(Xe, ye)
    extended_cluster_se = cluster_robust_se(extended_fit["info_matrix"], extended_fit["scores"], [o["match_id"] for o in fit_plus_tune])

    stale_test = [o for o in test_obs if o["stale"] is True]
    nonstale_test = [o for o in test_obs if o["stale"] is False]
    unknown_test = [o for o in test_obs if o["stale"] is None]

    def evaluate(model_key: str, beta: list[float]) -> dict:
        overall_ll, n_overall = _subset_log_loss(model_key, test_obs, beta, best_kappa_map, best_kp, best_ka)
        stale_ll, n_stale = _subset_log_loss(model_key, stale_test, beta, best_kappa_map, best_kp, best_ka)
        nonstale_ll, n_nonstale = _subset_log_loss(model_key, nonstale_test, beta, best_kappa_map, best_kp, best_ka)
        return {
            "n": n_overall,
            "log_loss_nats": overall_ll,
            "perplexity": perplexity(overall_ll) if overall_ll is not None else None,
            "n_stale": n_stale,
            "log_loss_nats_stale": stale_ll,
            "perplexity_stale": perplexity(stale_ll) if stale_ll is not None else None,
            "n_nonstale": n_nonstale,
            "log_loss_nats_nonstale": nonstale_ll,
            "perplexity_nonstale": perplexity(nonstale_ll) if nonstale_ll is not None else None,
        }

    global_eval = evaluate("global", global_fit["beta"])
    baseline_eval = evaluate("baseline", baseline_fit["beta"])
    extended_eval = evaluate("extended", extended_fit["beta"])

    n_stale_matches_approx = len({o["match_id"] for o in stale_test})
    n_nonstale_matches_approx = len({o["match_id"] for o in nonstale_test})

    return {
        "n_total_observations": len(observations),
        "n_fit": len(fit_obs),
        "n_tune": len(tune_obs),
        "n_test": len(test_obs),
        "n_stale_test": len(stale_test),
        "n_nonstale_test": len(nonstale_test),
        "n_unknown_staleness_test": len(unknown_test),
        "power_caveat": (
            f"Test slice: {len(test_obs)} map observations across ~{n_stale_matches_approx + n_nonstale_matches_approx} "
            f"clustered matches ({n_stale_matches_approx} stale-subset matches, {n_nonstale_matches_approx} "
            "nonstale-subset matches). This can reliably detect only a large effect (order ~0.05+ nats); "
            "a null result here should be read as 'underpowered to detect a small or moderate effect', "
            "not as 'no effect exists'. Pre-committing to this reading regardless of which way the numbers land."
        ),
        "kappa_map_grid_scores": kappa_map_scores,
        "best_kappa_map": best_kappa_map,
        "kappa_agent_grid_scores": agent_kappa_scores,
        "best_kappa_player": best_kp,
        "best_kappa_agent_map": best_ka,
        "n_hyperparameter_specs_tried": len(KAPPA_MAP_GRID) + len(KAPPA_PLAYER_GRID) * len(KAPPA_AGENT_MAP_GRID),
        "global_strength_only_diagnostic": {
            "description": "Model 0: team global win-rate diff only, no map identity. Calibrates what baseline's log loss means.",
            "beta": global_fit["beta"][0],
            **global_eval,
        },
        "baseline": {
            "beta_map_winrate_diff": baseline_fit["beta"][0],
            "beta_map_winrate_diff_cluster_se": baseline_cluster_se[0],
            "converged": baseline_fit["converged"],
            **baseline_eval,
        },
        "extended": {
            "beta_map_winrate_diff": extended_fit["beta"][0],
            "beta_map_winrate_diff_cluster_se": extended_cluster_se[0],
            "beta_agent_strength_diff": extended_fit["beta"][1],
            "beta_agent_strength_diff_cluster_se": extended_cluster_se[1],
            "converged": extended_fit["converged"],
            **extended_eval,
        },
        "improvement_nats_overall": (
            baseline_eval["log_loss_nats"] - extended_eval["log_loss_nats"]
            if baseline_eval["log_loss_nats"] is not None and extended_eval["log_loss_nats"] is not None
            else None
        ),
        "improvement_nats_stale": (
            baseline_eval["log_loss_nats_stale"] - extended_eval["log_loss_nats_stale"]
            if baseline_eval["log_loss_nats_stale"] is not None and extended_eval["log_loss_nats_stale"] is not None
            else None
        ),
        "improvement_nats_nonstale": (
            baseline_eval["log_loss_nats_nonstale"] - extended_eval["log_loss_nats_nonstale"]
            if baseline_eval["log_loss_nats_nonstale"] is not None and extended_eval["log_loss_nats_nonstale"] is not None
            else None
        ),
    }
