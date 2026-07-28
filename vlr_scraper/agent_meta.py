"""Agent pick-rate data: by player/team (from the global /stats page) and by
map (from each event's /event/agents/{id} page -- see parse_agent_pick_rates
in parse.py for why "rating per agent per map" specifically isn't included;
that's a separate, harder problem this module doesn't attempt).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from .fetch import Fetcher
from .parse import parse_agent_pick_rates, parse_events_list, parse_stats_table

PLAYER_STATS_PATH = Path("data/agent_meta/player_stats.jsonl")
MAP_PICK_RATES_PATH = Path("data/agent_meta/map_pick_rates.jsonl")


def collect_player_agent_stats(pages: int = 5, min_rounds: int = 100, verbose: bool = True) -> dict:
    """Fetch the global /stats leaderboard (tier=all, region=all, span=all)
    across `pages` pages of up to 100 players each. Each row already carries
    that player's own per-agent pick-rate breakdown (parse_stats_table's
    `agents` field) -- this is the raw material for both "pick rate by
    player" (used directly) and "pick rate by team" (aggregated below).
    """
    PLAYER_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []
    with Fetcher() as fetcher:
        for page in range(1, pages + 1):
            html = fetcher.get_html(
                "/stats",
                params={"tier": "all", "region": "all", "span": "all", "min_rounds": str(min_rounds), "page": str(page)},
            )
            rows = parse_stats_table(html, source=f"global:page{page}", filters={})
            if not rows:
                break
            all_rows.extend(rows)
            if verbose:
                print(f"  page {page}: {len(rows)} players", file=sys.stderr)

    with PLAYER_STATS_PATH.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r.to_dict()) + "\n")

    return {"players": len(all_rows), "pages_fetched": page}


def player_agent_pick_rates() -> list[dict]:
    """One row per (player, agent): that player's pick rate on that agent,
    straight from the stats table -- no aggregation needed."""
    rows = [json.loads(l) for l in PLAYER_STATS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    for r in rows:
        for a in r["agents"]:
            out.append(
                {
                    "player_id": r["player_id"],
                    "player_alias": r["player_alias"],
                    "team": r["team_or_country"],
                    "agent": a["agent"],
                    "pick_rate_pct": a["pct"],
                    "player_maps": r["maps"],
                    "player_rating": r["rating"],
                }
            )
    return out


def team_agent_pick_rates() -> list[dict]:
    """One row per (team, agent): pick rate aggregated across that team's
    players, weighted by each player's maps played (so a 5-map bench player
    doesn't count as much as the starting lineup)."""
    rows = [json.loads(l) for l in PLAYER_STATS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    # team -> agent -> [weighted_pct_sum, weight_sum]
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    for r in rows:
        team = r["team_or_country"]
        weight = r["maps"] or 0
        if not team or not weight:
            continue
        for a in r["agents"]:
            acc[team][a["agent"]][0] += (a["pct"] or 0) * weight
            acc[team][a["agent"]][1] += weight

    out = []
    for team, agents in acc.items():
        for agent, (weighted_sum, weight_sum) in agents.items():
            if weight_sum > 0:
                out.append({"team": team, "agent": agent, "pick_rate_pct": weighted_sum / weight_sum, "total_maps": weight_sum})
    return out


def collect_map_pick_rates(max_events: int = 40, event_pages: int = 3, verbose: bool = True) -> dict:
    """Fetch recent completed events' /event/agents/{id} pages (pick rate per
    agent per map, per event -- see parse_agent_pick_rates)."""
    existing_ids: set[int] = set()
    if MAP_PICK_RATES_PATH.exists():
        for line in MAP_PICK_RATES_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing_ids.add(json.loads(line)["event_id"])

    MAP_PICK_RATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = {"already_had": len(existing_ids), "added": 0, "no_data": 0}

    with Fetcher() as fetcher:
        events: list[dict] = []
        for page in range(1, event_pages + 1):
            html = fetcher.get_html("/events", params={"page": str(page)})
            events.extend(parse_events_list(html))

        completed = [e for e in events if e["status"] == "completed" and e["event_id"] not in existing_ids]
        completed = completed[:max_events]

        with MAP_PICK_RATES_PATH.open("a", encoding="utf-8") as out:
            for i, ev in enumerate(completed, 1):
                html = fetcher.get_html(f"/event/agents/{ev['event_id']}", params={"map_id": "all"})
                entries = parse_agent_pick_rates(html, event_id=ev["event_id"])
                if not entries:
                    summary["no_data"] += 1
                    continue
                for entry in entries:
                    row = entry.to_dict()
                    row["event_name"] = ev["name"]
                    out.write(json.dumps(row) + "\n")
                summary["added"] += 1
                if verbose and i % 10 == 0:
                    print(f"  [{i}/{len(completed)}] events processed", file=sys.stderr)

    return summary


def map_agent_pick_rates() -> list[dict]:
    """Aggregated across all collected events: for each (map, agent), average
    pick rate weighted by games played in that event's row."""
    if not MAP_PICK_RATES_PATH.exists():
        return []
    rows = [json.loads(l) for l in MAP_PICK_RATES_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    acc: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        if r["map_name"] == "All" or not r["games"] or r["pick_rate_pct"] is None:
            continue
        key = (r["map_name"], r["agent"])
        acc[key][0] += r["pick_rate_pct"] * r["games"]
        acc[key][1] += r["games"]
    out = []
    for (map_name, agent), (weighted_sum, weight_sum) in acc.items():
        if weight_sum > 0:
            out.append({"map": map_name, "agent": agent, "pick_rate_pct": weighted_sum / weight_sum, "total_games": weight_sum})
    return out
