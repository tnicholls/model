"""Raw JSON persistence + flattening into tabular rows for modeling."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .models import AgentMapPickRate, BetLine, MatchSummary, PlayerStat, RosterPlayer, StandingEntry, TeamInfo


def save_json(data, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def player_stats_to_rows(stats: list[PlayerStat]) -> list[dict]:
    """Flatten PlayerStat rows into one row per player (agents joined into a string)."""
    rows = []
    for s in stats:
        d = s.to_dict()
        agents = d.pop("agents")
        d["agents"] = "; ".join(f"{a['agent']}:{a['pct']}" for a in agents)
        d["filters"] = json.dumps(d["filters"], separators=(",", ":"))
        rows.append(d)
    return rows


def standings_to_rows(entries: list[StandingEntry]) -> list[dict]:
    return [e.to_dict() for e in entries]


def matches_to_rows(matches: list[MatchSummary]) -> list[dict]:
    return [m.to_dict() for m in matches]


def odds_to_rows(lines: list[BetLine]) -> list[dict]:
    return [b.to_dict() for b in lines]


def agent_pick_rates_to_rows(entries: list[AgentMapPickRate]) -> list[dict]:
    return [e.to_dict() for e in entries]


def roster_to_rows(team: TeamInfo) -> list[dict]:
    """One row per roster player, with team fields repeated (long format)."""
    rows = []
    for p in team.roster + team.staff:
        row = {
            "team_id": team.team_id,
            "team_name": team.name,
            "team_tag": team.tag,
            "team_country": team.country,
        }
        row.update(p.to_dict())
        rows.append(row)
    return rows
