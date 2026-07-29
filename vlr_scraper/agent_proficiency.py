"""Agent-proficiency map-strength pipeline: builds the factorised map-strength
estimate

    r[t,m] = r[t] + sum_p pi[p, a(p,m)] * v[a(p,m), m]

as an alternative to the direct per-team-map win rate already used elsewhere
in this codebase (see veto_model.shrunk_rate). Spec: see the build-spec
message this module was written against (agent/character map-proficiency
work, July 2026).

This module covers the first two pipeline stages:
  1. collect_player_map_stats  -- scrape per-(match,map,player) agent+rating
  2. build_patch_dates / build_map_rotation -- reference tables, both derived
     from already-collected data rather than hand-authored (see docstrings)

v[a,m], pi[p,a], and the Step 3-4 modal-comp -> agent_strength[t,m]
projection live in base_model.py instead, built as one incremental
walk-forward pass alongside the base model's own features -- see that
module's docstring for why.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from .fetch import Fetcher
from .parse import parse_match_player_map_stats
from .veto import VETO_DATASET_PATH

PLAYER_MAP_STATS_PATH = Path("data/agent_proficiency/player_map_stats.jsonl")
PATCH_DATES_PATH = Path("data/agent_proficiency/patch_dates.json")
MAP_ROTATION_PATH = Path("data/agent_proficiency/map_rotation.json")

# Pre-registered constants (fixed before any test was run -- see build spec).
#
# Revised after the first run: the spec's original criterion (games since the most recent
# major patch) is collinear with calendar time -- a chronological last-20% test slice can
# land entirely inside one recent patch, making 100% of it "stale" with no fresh comparison
# group to test the conditional hypothesis against (confirmed: it did, exactly this). Switched
# the primary criterion to raw team-map sample count, which varies within any time window
# regardless of patch boundaries -- and is closer to the actual mechanism anyway (the agent
# factorisation should help most where the *direct* team-map estimate has the least data,
# patch-related or not). Threshold moved from <5 to <2: median team-map history at any point
# in this dataset is just 1 prior game (700 matches spread thin across ~249 teams x 9 maps),
# so <5 was true for ~97% of observations everywhere, not just the test slice -- a threshold
# problem independent of the collinearity one. <2 gives a usable ~65/35 stale/nonstale split.
# This is a structural fix to a criterion that produced a degenerate (0% or 100%) split, not a
# search for significance -- see agent_proficiency_report's determinism/robustness notes.
STALE_MIN_TEAM_MAP_GAMES = 2  # fewer than this many total prior team-map games (either team) -> stale
STALE_ROTATION_WINDOW_DAYS = 90  # map (re)entered the active pool within this many days -> stale (secondary criterion)
# Widened after the first run: both hit the original grid's edge (kappa_player=160,
# kappa_agent_map=2) -- per spec, "if a tuned kappa lands at the edge of its grid,
# widen the grid and rerun." (Widening the search itself isn't a significance-chasing
# adjustment -- nothing about the stale-subset definition or evaluation slice changes.)
KAPPA_PLAYER_GRID = (0.5, 1, 2, 5, 10, 20, 40, 80, 160, 320, 640, 1280)
KAPPA_AGENT_MAP_GRID = (0.25, 0.5, 1, 2, 5, 10, 20, 40, 80, 160, 320)
RANDOM_SEED = 20260727


# --------------------------------------------------------------------------
# Stage 1: scraping
# --------------------------------------------------------------------------


def collect_player_map_stats(verbose: bool = True) -> dict:
    """Walk every match already in VETO_DATASET_PATH (its match_url, team1_id,
    team2_id are reused as-is -- no need to re-derive them from the page) and
    scrape its per-map player/agent/rating rows. Skips matches already
    present in PLAYER_MAP_STATS_PATH. Separate JSONL, existing files
    untouched.
    """
    if not VETO_DATASET_PATH.exists():
        raise FileNotFoundError(f"{VETO_DATASET_PATH} does not exist -- run veto.collect_veto_dataset() first")

    matches = [json.loads(l) for l in VETO_DATASET_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]

    existing_ids: set[int] = set()
    if PLAYER_MAP_STATS_PATH.exists():
        for line in PLAYER_MAP_STATS_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing_ids.add(json.loads(line)["match_id"])

    todo = [m for m in matches if m["match_id"] not in existing_ids]
    summary = {"already_had": len(existing_ids), "to_fetch": len(todo), "added": 0, "no_data": 0, "failed": 0}
    if not todo:
        return summary

    PLAYER_MAP_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with Fetcher() as fetcher:
        with PLAYER_MAP_STATS_PATH.open("a", encoding="utf-8") as out:
            for i, m in enumerate(todo, 1):
                try:
                    html = fetcher.get_html(m["match_url"])
                    rows = parse_match_player_map_stats(html, match_id=m["match_id"])
                except Exception as e:  # noqa: BLE001 -- one bad match page shouldn't kill the batch
                    summary["failed"] += 1
                    if verbose:
                        print(f"  failed match {m['match_id']}: {e}", file=sys.stderr)
                    continue
                if not rows:
                    summary["no_data"] += 1
                    continue
                for r in rows:
                    out.write(json.dumps(r.to_dict()) + "\n")
                summary["added"] += 1
                if verbose and i % 25 == 0:
                    print(f"  [{i}/{len(todo)}] matches processed", file=sys.stderr)
    return summary


def load_date_by_match() -> dict[int, str]:
    """match_id -> earliest known match_date_utc, from player_map_stats."""
    if not PLAYER_MAP_STATS_PATH.exists():
        return {}
    out: dict[int, str] = {}
    for line in PLAYER_MAP_STATS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["match_date_utc"]:
            out.setdefault(r["match_id"], r["match_date_utc"])
    return out


def load_patch_by_match() -> dict[int, str]:
    if not PLAYER_MAP_STATS_PATH.exists():
        return {}
    out: dict[int, str] = {}
    for line in PLAYER_MAP_STATS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["patch"]:
            out.setdefault(r["match_id"], r["patch"])
    return out


def load_matches_ordered() -> list[dict]:
    """VETO_DATASET_PATH matches sorted by (match_date_utc, match_id) -- the
    same canonical chronological order used throughout agent_proficiency.py
    and base_model.py. Falls back to match_id-only ordering (the proxy used
    elsewhere in this codebase) for the rare match with no recovered date.
    """
    if not VETO_DATASET_PATH.exists():
        raise FileNotFoundError(f"{VETO_DATASET_PATH} does not exist -- run veto.collect_veto_dataset() first")
    matches = [json.loads(l) for l in VETO_DATASET_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    dates = load_date_by_match()
    matches.sort(key=lambda m: (dates.get(m["match_id"], ""), m["match_id"]))
    return matches


# --------------------------------------------------------------------------
# Stage 2: reference tables (both derived from already-collected data, not
# hand-authored -- see each function's docstring for why)
# --------------------------------------------------------------------------


def build_patch_dates() -> dict:
    """Patch -> first-seen-date, built from the `patch` field scraped directly
    off each match page (see parse.parse_match_header_date_patch) rather than
    hand-maintained: vlr.gg already stamps every match with its patch
    version, so this is exact for the matches in the dataset (as opposed to
    reconstructed from memory of the patch-release calendar, which would be
    both stale past this codebase's knowledge cutoff and approximate even
    within it). "First seen" slightly *overestimates* the true release date
    only for the very first patch in the window (whose true start predates
    the earliest scraped match) -- immaterial here since that patch is never
    itself the "most recent major patch" reference point for staleness.
    """
    if not PLAYER_MAP_STATS_PATH.exists():
        raise FileNotFoundError(f"{PLAYER_MAP_STATS_PATH} does not exist -- run collect_player_map_stats() first")
    first_seen: dict[str, str] = {}
    for line in PLAYER_MAP_STATS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        patch, date = r["patch"], r["match_date_utc"]
        if not patch or not date:
            continue
        if patch not in first_seen or date < first_seen[patch]:
            first_seen[patch] = date
    ordered = dict(sorted(first_seen.items(), key=lambda kv: kv[1]))
    PATCH_DATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    PATCH_DATES_PATH.write_text(json.dumps(ordered, indent=2), encoding="utf-8")
    return ordered


def most_recent_patch_at(date: str, patch_dates: dict[str, str]) -> str | None:
    """The patch in effect on `date`: the patch with the latest first-seen
    date that is <= `date`. Sorted comparison on the "YYYY-MM-DD..." string
    form works directly (no date parsing needed)."""
    candidates = sorted((d, p) for p, d in patch_dates.items() if d <= date)
    return candidates[-1][1] if candidates else None


def build_map_rotation() -> dict:
    """Map -> list of {entered_pool_date, source} spans, derived empirically
    from which maps actually appear in each match's veto pool over time
    (data/veto/matches.jsonl), rather than hand-authored from memory of the
    map-pool rotation calendar. A map "enters" the pool at the first match
    date where it appears after a gap of more than 21 days since it last
    appeared in any pool (21 days is comfortably longer than any realistic
    gap between matches of an in-pool map, and comfortably shorter than a
    real rotation-out period) -- so a map continuously in rotation gets one
    entry, and a map that rotates out and back in gets two.
    """
    if not VETO_DATASET_PATH.exists():
        raise FileNotFoundError(f"{VETO_DATASET_PATH} does not exist -- run veto.collect_veto_dataset() first")
    matches = [json.loads(l) for l in VETO_DATASET_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]

    # Need a date per match; veto dataset only has date_display (not sortable
    # UTC), so join against player_map_stats' match_date_utc where available.
    date_by_match: dict[int, str] = {}
    if PLAYER_MAP_STATS_PATH.exists():
        for line in PLAYER_MAP_STATS_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["match_date_utc"]:
                date_by_match.setdefault(r["match_id"], r["match_date_utc"])

    appearances: dict[str, list[str]] = defaultdict(list)
    for m in matches:
        date = date_by_match.get(m["match_id"])
        if not date:
            continue
        pool = {s["map"] for s in m["veto"] if s.get("map")}
        for mp in pool:
            appearances[mp].append(date)

    GAP_DAYS = 21
    rotation: dict[str, list[dict]] = {}
    for mp, dates in appearances.items():
        dates = sorted(dates)
        spans = []
        prev = None
        for d in dates:
            if prev is None or _days_between(prev, d) > GAP_DAYS:
                spans.append({"entered_pool_date": d, "source": "derived_from_veto_pool"})
            prev = d
        rotation[mp] = spans

    MAP_ROTATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    MAP_ROTATION_PATH.write_text(json.dumps(rotation, indent=2), encoding="utf-8")
    return rotation


def _days_between(date_a: str, date_b: str) -> float:
    """Days between two "YYYY-MM-DD HH:MM:SS"-ish strings, date part only."""
    from datetime import date as _date

    def parse(d: str) -> _date:
        y, mo, da = d[:10].split("-")
        return _date(int(y), int(mo), int(da))

    return (parse(date_b) - parse(date_a)).days


def most_recent_pool_entry_at(map_name: str, date: str, rotation: dict[str, list[dict]]) -> str | None:
    """Latest entered_pool_date for `map_name` that is <= `date`."""
    spans = rotation.get(map_name, [])
    candidates = sorted(s["entered_pool_date"] for s in spans if s["entered_pool_date"] <= date)
    return candidates[-1] if candidates else None


# Stage 3 (v[a,m], pi[p,a]) and Stage 4 (modal comp -> agent_strength[t,m])
# are NOT here: they need to be point-in-time-consistent with the same
# team-map win-rate bookkeeping the base model uses for its own feature, and
# efficient enough to re-evaluate under many (kappa_player, kappa_agent_map)
# combinations during tuning. An earlier version of this module implemented
# them as standalone as-of-index functions (recomputing from scratch per
# query); base_model.py's single incremental walk-forward pass over shared
# state is both more correct (one pass, guaranteed consistent snapshots for
# both the map-observation and veto-decision datasets) and far cheaper (no
# re-scanning history per query) -- see base_model.build_walkforward,
# _agent_terms_for_team, and _standardized_agent_strength.
