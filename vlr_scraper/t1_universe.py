"""T1-team-first data collection, replacing the earlier global-results-feed
scrape (veto.collect_veto_dataset over `/matches/results`, most-recent-700).
That approach mixed VCT International with Challengers/Game Changers/
collegiate matches of wildly different team quality, and even restricted to
one tier, "most recent N matches" spreads too thin: median team-map history
across the whole first dataset was 1 prior game. Team-first data collection
is the actual fix; "last 10 matches per team" is not enough on its own --
10 matches over ~9 maps is ~3-4 map appearances per team, i.e. the same
starved regime with fewer teams. ~30-40 matches per team is what it takes
for a team-map cell to reach ~8-12 games.

Pipeline:
  1. discover_t1_teams -- the current VCT International franchise list for
     four regions (Americas, EMEA, Pacific, China), from each region's live
     regular-season event pages (not vlr.gg's general Elo rankings, which
     mixes in non-franchised teams under the same geographic buckets).
  2. collect_t1_match_ids -- for every team, walk its own /team/matches
     history (time floor + minimum-count, not a fixed "last N"). A team's
     own match history already includes cross-region events (Masters,
     Champions) it played in -- these are kept, not filtered to
     region-league matches only, because they're what connects the four
     regions into one comparable strength scale instead of four
     disconnected graphs.
  3. collect_t1_match_details -- fetch full match detail (veto, per-map
     results, team ids) for the deduplicated match set, writing to the SAME
     paths veto.py/agent_proficiency.py already read
     (veto.VETO_DATASET_PATH, veto.TEAM_TAGS_PATH) -- this is a deliberate
     replacement of the earlier dataset (superseded, not supplemented), so
     every downstream module (veto_model, base_model, agent_proficiency)
     works unchanged once this has run.
  4. team_map_density_report -- the required check: distribution of
     (team, map) game counts, run BEFORE refitting anything. If the median
     doesn't move from ~1 to ~8-12, the scrape didn't fix the problem.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

from bs4 import BeautifulSoup

from .fetch import Fetcher
from .parse import _clean, parse_event_team_links, parse_team_matches_list
from .veto import VETO_DATASET_PATH, _TEAM_HREF_RE, parse_match_maps, parse_veto_note

T1_TEAMS_PATH = Path("data/t1/teams.json")
T1_MATCH_IDS_PATH = Path("data/t1/match_ids.json")

# VCT International 2026 regular-season event IDs per region (Stage 1 + Stage 2 -- both,
# so a team that hasn't played a Stage 2 match yet is still counted). Season-specific;
# will need updating next season. Rediscover via parse.parse_events_list() filtered to
# names matching "VCT <year>: <Region> Stage *".
REGION_EVENTS = {
    "americas": [2860, 2977],
    "emea": [2863, 2976],
    "pacific": [2775, 2776],
    "china": [2864, 2978],
}

MIN_MATCHES_PER_TEAM = 30
TIME_FLOOR_DAYS = 365
MAX_PAGES_PER_TEAM = 6


def discover_t1_teams(region_events: dict[str, list[int]] = REGION_EVENTS, verbose: bool = True) -> dict[str, dict[str, str]]:
    """{region: {team_id_str: name}} for every team appearing on any of that
    region's configured event pages. Every team link on the page, not just
    the standings/bracket table, so an in-progress group stage's not-yet-
    placed teams are still counted."""
    result: dict[str, dict[str, str]] = {}
    with Fetcher() as fetcher:
        for region, event_ids in region_events.items():
            teams: dict[int, str] = {}
            for eid in event_ids:
                html = fetcher.get_html(f"/event/{eid}")
                teams.update(parse_event_team_links(html))
            result[region] = {str(tid): name for tid, name in teams.items()}
            if verbose:
                print(f"  {region}: {len(teams)} teams", file=sys.stderr)
    T1_TEAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    T1_TEAMS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def collect_t1_match_ids(teams: dict[str, dict[str, str]] | None = None, verbose: bool = True) -> dict:
    """For every T1 team, walk its /team/matches page (paginated) collecting
    completed match_ids until both MIN_MATCHES_PER_TEAM is reached and the
    TIME_FLOOR_DAYS boundary has been passed, or MAX_PAGES_PER_TEAM runs out.
    If the floor-restricted set is short of the minimum (inactive team),
    tops up with its most recent matches regardless of the floor -- a
    thinner-than-ideal team is better than an artificially excluded one.
    """
    if teams is None:
        teams = json.loads(T1_TEAMS_PATH.read_text(encoding="utf-8")) if T1_TEAMS_PATH.exists() else discover_t1_teams()

    floor_date = (date.today() - timedelta(days=TIME_FLOOR_DAYS)).isoformat()
    all_matches: dict[int, dict] = {}  # match_id -> {match_url, team_ids: set}
    per_team_counts: dict[str, int] = {}

    with Fetcher() as fetcher:
        for region, team_map in teams.items():
            for team_id_str, name in team_map.items():
                team_id = int(team_id_str)
                collected: list[dict] = []
                page = 1
                while page <= MAX_PAGES_PER_TEAM:
                    html = fetcher.get_html(f"/team/matches/{team_id}/team", params={"page": page})
                    rows = parse_team_matches_list(html)
                    completed = [r for r in rows if r["status"] == "Completed" and r["date"]]
                    if not completed:
                        break
                    collected.extend(completed)
                    oldest = min(r["date"] for r in completed)
                    if len(collected) >= MIN_MATCHES_PER_TEAM and oldest < floor_date:
                        break
                    page += 1

                within_floor = [r for r in collected if r["date"] >= floor_date]
                use = within_floor if len(within_floor) >= MIN_MATCHES_PER_TEAM else collected[:MIN_MATCHES_PER_TEAM]
                per_team_counts[f"{region}:{name}"] = len(use)
                for r in use:
                    entry = all_matches.setdefault(r["match_id"], {"match_url": r["match_url"], "team_ids": set()})
                    entry["team_ids"].add(team_id)
                if verbose:
                    print(f"  {region} {name} ({team_id}): {len(use)} matches (scanned {len(collected)})", file=sys.stderr)

    out = {str(mid): {"match_url": v["match_url"], "team_ids": sorted(v["team_ids"])} for mid, v in all_matches.items()}
    T1_MATCH_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    T1_MATCH_IDS_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return {
        "n_teams": sum(len(t) for t in teams.values()),
        "n_unique_matches": len(all_matches),
        "per_team_counts": per_team_counts,
    }


def _parse_team_ids_and_names(html: str) -> tuple[int | None, str | None, int | None, str | None]:
    soup = BeautifulSoup(html, "lxml")
    header = soup.find("div", class_="match-header-vs")
    if header is None:
        return None, None, None, None
    ids: list[int | None] = [None, None]
    names: list[str | None] = [None, None]
    for link in header.find_all("a", class_="match-header-link"):
        classes = link.get("class") or []
        idx = 0 if "mod-1" in classes else (1 if "mod-2" in classes else None)
        if idx is None:
            continue
        m = _TEAM_HREF_RE.search(link.get("href", ""))
        ids[idx] = int(m.group(1)) if m else None
        name_div = link.find(class_="wf-title-med") or link.find(class_="text-of")
        names[idx] = _clean(name_div.get_text()) if name_div else None
    return ids[0], names[0], ids[1], names[1]


def collect_t1_match_details(match_entries: dict[str, dict] | None = None, verbose: bool = True) -> dict:
    """Fetch full match detail for every T1 match_id and write it to
    VETO_DATASET_PATH in the SAME row schema veto.collect_veto_dataset uses
    -- a deliberate replacement of that dataset's contents, not an addition,
    so every downstream module keeps working unchanged. Call
    veto.build_team_tag_cache() and agent_proficiency.collect_player_map_stats()
    after this to finish rebuilding the full pipeline's inputs.
    """
    if match_entries is None:
        match_entries = json.loads(T1_MATCH_IDS_PATH.read_text(encoding="utf-8"))

    VETO_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = {"total": len(match_entries), "written": 0, "no_veto_note": 0, "failed": 0}

    with Fetcher() as fetcher:
        with VETO_DATASET_PATH.open("w", encoding="utf-8") as out:
            for i, (match_id_str, entry) in enumerate(sorted(match_entries.items(), key=lambda kv: int(kv[0])), 1):
                match_id = int(match_id_str)
                try:
                    html = fetcher.get_html(entry["match_url"])
                    soup = BeautifulSoup(html, "lxml")
                    note_div = soup.find("div", class_="match-header-note")
                    note_text = _clean(note_div.get_text(" ")) if note_div else None
                    veto = [s.to_dict() for s in parse_veto_note(note_text)] if note_text else []
                    maps = [r.to_dict() for r in parse_match_maps(html)]
                    team1_id, team1_name, team2_id, team2_name = _parse_team_ids_and_names(html)

                    event_div = soup.find(class_="match-header-event")
                    event_name = re.sub(r"\s+", " ", event_div.get_text(" ", strip=True)) if event_div else None
                    date_div = soup.find("div", class_="match-header-date")
                    date_display = None
                    if date_div:
                        moment = date_div.find(class_="moment-tz-convert")
                        date_display = _clean(moment.get_text()) if moment else None

                    winner = None
                    if team1_name and team2_name:
                        t1_wins = sum(1 for mr in maps if mr["winner"] == team1_name)
                        t2_wins = sum(1 for mr in maps if mr["winner"] == team2_name)
                        if t1_wins != t2_wins:
                            winner = team1_name if t1_wins > t2_wins else team2_name

                    row = {
                        "match_id": match_id,
                        "match_url": entry["match_url"],
                        "date_display": date_display,
                        "event_name": event_name,
                        "team1": team1_name,
                        "team2": team2_name,
                        "team1_id": team1_id,
                        "team2_id": team2_id,
                        "winner": winner,
                        "veto_note_raw": note_text,
                        "veto": veto,
                        "maps": maps,
                    }
                    out.write(json.dumps(row) + "\n")
                    summary["written"] += 1
                    if not veto:
                        summary["no_veto_note"] += 1
                except Exception as e:  # noqa: BLE001 -- one bad match page shouldn't kill the batch
                    summary["failed"] += 1
                    if verbose:
                        print(f"  failed match {match_id}: {e}", file=sys.stderr)
                if verbose and i % 50 == 0:
                    print(f"  [{i}/{len(match_entries)}] matches fetched", file=sys.stderr)

    return summary


def team_map_density_report(veto_dataset_path: Path = VETO_DATASET_PATH) -> dict:
    """The required check before refitting anything: distribution of
    (team, map) completed-game counts. Report median/mean/min/max -- if the
    median hasn't moved from the old dataset's ~1 to something like 8-12,
    the team-first scrape didn't fix the density problem."""
    rows = [json.loads(l) for l in veto_dataset_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    counts: dict[tuple[int, str], int] = defaultdict(int)
    for r in rows:
        t1_id, t2_id = r.get("team1_id"), r.get("team2_id")
        for mr in r["maps"]:
            if mr["winner"] is None:
                continue
            if t1_id is not None:
                counts[(t1_id, mr["map_name"])] += 1
            if t2_id is not None:
                counts[(t2_id, mr["map_name"])] += 1
    vals = sorted(counts.values())
    n = len(vals)
    if n == 0:
        return {"n_team_map_cells": 0}
    median = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    return {
        "n_team_map_cells": n,
        "min": vals[0],
        "median": median,
        "mean": sum(vals) / n,
        "max": vals[-1],
        "n_matches": len(rows),
        "n_completed_maps": sum(1 for r in rows for mr in r["maps"] if mr["winner"] is not None),
    }
