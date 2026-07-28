"""Map veto parsing + a walk-forward test of a naive "ban your worst map" rule.

vlr.gg match pages carry the full veto sequence as a single plain-text note
(e.g. "LOUD ban Split; SEN ban Sunset; LOUD pick Lotus; SEN pick Breeze;
LOUD ban Haven; SEN ban Summit; Ascent remains") plus per-map results in the
per-game stat blocks. Combining the two across many matches lets us check
whether a simple team-map win-rate model would have predicted each team's
first ban -- a cheap "is there exploitable structure here at all" test
before building anything fancier.

Two things this module is careful about:

  * Team identity. The veto note uses each team's short tag (e.g. "SEN"),
    while match results use the full name (e.g. "Sentinels") -- and tags are
    not reliably derivable from the name by string similarity (e.g. Paper
    Rex's tag is "PRX", which shares more characters with "DRX" than with
    "Paper Rex" itself). Resolving this correctly means fetching each team's
    actual tag from its vlr.gg team page rather than guessing.

  * Lookahead. A team's map win rate must only be computed from matches that
    happened *before* the one being tested, or the test is circular (it'd
    "know" the outcome it's trying to predict). Match IDs on vlr.gg are
    assigned sequentially over time, so sorting by match_id is used as a
    cheap proxy for chronological order.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from bs4 import BeautifulSoup

from .fetch import Fetcher
from .models import MatchSummary
from .parse import _clean, _direct_text, parse_matches, parse_team

VETO_DATASET_PATH = Path("data/veto/matches.jsonl")
TEAM_TAGS_PATH = Path("data/veto/team_tags.json")

_VETO_ACTION_RE = re.compile(r"^(?P<team>.+?)\s+(?P<action>ban|pick)\s+(?P<map>.+)$")
_VETO_REMAINS_RE = re.compile(r"^(?P<map>.+?)\s+remains$")
_TEAM_HREF_RE = re.compile(r"/team/(\d+)/")


@dataclass
class VetoStep:
    team: str | None  # None for the decider
    action: str  # "ban" | "pick" | "decider"
    map: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MapResult:
    map_name: str
    team1: str
    team1_score: int | None
    team2: str
    team2_score: int | None
    winner: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def parse_veto_note(note: str) -> list[VetoStep]:
    """Parse a match-header-note string into an ordered list of veto steps."""
    steps: list[VetoStep] = []
    for token in note.split(";"):
        token = token.strip()
        if not token:
            continue
        remains_match = _VETO_REMAINS_RE.match(token)
        if remains_match:
            steps.append(VetoStep(team=None, action="decider", map=remains_match.group("map").strip()))
            continue
        action_match = _VETO_ACTION_RE.match(token)
        if action_match:
            steps.append(
                VetoStep(
                    team=action_match.group("team").strip(),
                    action=action_match.group("action"),
                    map=action_match.group("map").strip(),
                )
            )
    return steps


def parse_match_maps(html: str) -> list[MapResult]:
    """Parse per-map results from a match page's per-game stat blocks."""
    soup = BeautifulSoup(html, "lxml")
    results: list[MapResult] = []
    for game_div in soup.find_all("div", class_="vm-stats-game"):
        game_id = game_div.get("data-game-id")
        if not game_id or game_id == "all":
            continue
        header = game_div.find("div", class_="vm-stats-game-header")
        if header is None:
            continue
        team_divs = header.find_all("div", class_="team", recursive=False)
        if len(team_divs) < 2:
            continue

        map_div = header.find("div", class_="map")
        map_name = None
        if map_div:
            name_span = map_div.find("span", style=True)
            map_name = _direct_text(name_span) if name_span else _clean(map_div.get_text(" "))
            if map_name:
                map_name = map_name.split("\n")[0].strip()

        teams = []
        for td in team_divs:
            name_div = td.find("div", class_="team-name")
            score_div = td.find("div", class_="score")
            name = _clean(name_div.get_text()) if name_div else None
            score_text = _clean(score_div.get_text()) if score_div else None
            score = int(score_text) if score_text and score_text.isdigit() else None
            is_winner = bool(score_div and "mod-win" in (score_div.get("class") or []))
            teams.append((name, score, is_winner))

        if map_name and len(teams) == 2:
            (t1, s1, w1), (t2, s2, w2) = teams
            winner = t1 if w1 else (t2 if w2 else None)
            results.append(MapResult(map_name=map_name, team1=t1, team1_score=s1, team2=t2, team2_score=s2, winner=winner))
    return results


def _parse_team_ids(html: str) -> tuple[int | None, int | None]:
    """Team IDs (not names) for team1/team2, from the match header's team links."""
    soup = BeautifulSoup(html, "lxml")
    header = soup.find("div", class_="match-header-vs")
    if header is None:
        return None, None
    links = header.find_all("a", class_="match-header-link")
    ids: list[int | None] = [None, None]
    for link in links:
        classes = link.get("class") or []
        idx = 0 if "mod-1" in classes else (1 if "mod-2" in classes else None)
        if idx is None:
            continue
        m = _TEAM_HREF_RE.search(link.get("href", ""))
        ids[idx] = int(m.group(1)) if m else None
    return ids[0], ids[1]


def collect_veto_dataset(target_matches: int = 700, max_pages: int = 120, verbose: bool = True) -> dict:
    """Fetch recent completed matches (regardless of betting-odds coverage --
    veto/map data doesn't depend on which bookmakers sponsored the match) and
    save their veto note + per-map results to VETO_DATASET_PATH."""
    existing_ids: set[int] = set()
    if VETO_DATASET_PATH.exists():
        for line in VETO_DATASET_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing_ids.add(json.loads(line)["match_id"])

    summary = {"already_had": len(existing_ids), "added": 0, "no_veto_note": 0, "pages_scanned": 0}
    needed = target_matches - len(existing_ids)
    if needed <= 0:
        return summary

    VETO_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with Fetcher() as fetcher:
        with VETO_DATASET_PATH.open("a", encoding="utf-8") as out:
            page = 1
            while summary["added"] < needed and page <= max_pages:
                html = fetcher.get_html("/matches/results", params={"page": page})
                matches: list[MatchSummary] = parse_matches(html, source="global:results")
                summary["pages_scanned"] += 1
                if not matches:
                    break
                for m in matches:
                    if summary["added"] >= needed:
                        break
                    if m.status != "Completed" or m.match_id in existing_ids:
                        continue
                    match_html = fetcher.get_html(m.match_url)
                    soup = BeautifulSoup(match_html, "lxml")
                    note_div = soup.find("div", class_="match-header-note")
                    note_text = _clean(note_div.get_text(" ")) if note_div else None
                    veto = [s.to_dict() for s in parse_veto_note(note_text)] if note_text else []
                    maps = [r.to_dict() for r in parse_match_maps(match_html)]
                    team1_id, team2_id = _parse_team_ids(match_html)

                    row = {
                        "match_id": m.match_id,
                        "match_url": m.match_url,
                        "date_display": m.date_display,
                        "event_name": m.event_name,
                        "team1": m.team1,
                        "team2": m.team2,
                        "team1_id": team1_id,
                        "team2_id": team2_id,
                        "winner": m.team1 if m.team1_won else (m.team2 if m.team2_won else None),
                        "veto_note_raw": note_text,
                        "veto": veto,
                        "maps": maps,
                    }
                    out.write(json.dumps(row) + "\n")
                    existing_ids.add(m.match_id)
                    summary["added"] += 1
                    if not veto:
                        summary["no_veto_note"] += 1
                    if verbose and summary["added"] % 25 == 0:
                        print(f"  [{summary['added']}/{needed}] match {m.match_id}", file=sys.stderr)
                page += 1
    return summary


def build_team_tag_cache(verbose: bool = True) -> dict:
    """Fetch each team's actual tag (e.g. team_id 2 -> "SEN") from its vlr.gg
    team page, for every team appearing in VETO_DATASET_PATH. Tags are not
    reliably derivable from team names by string similarity, so this is the
    authoritative source used to match veto-note labels to team1/team2.
    Cached to TEAM_TAGS_PATH; only fetches teams not already cached.
    """
    if not VETO_DATASET_PATH.exists():
        raise FileNotFoundError(f"{VETO_DATASET_PATH} does not exist -- run collect_veto_dataset() first")

    cache: dict[str, str] = {}
    if TEAM_TAGS_PATH.exists():
        cache = json.loads(TEAM_TAGS_PATH.read_text(encoding="utf-8"))

    team_ids: set[int] = set()
    for line in VETO_DATASET_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for tid in (row.get("team1_id"), row.get("team2_id")):
            if tid is not None:
                team_ids.add(tid)

    to_fetch = sorted(tid for tid in team_ids if str(tid) not in cache)
    summary = {"already_cached": len(cache), "fetched": 0, "failed": 0}
    if not to_fetch:
        return summary

    TEAM_TAGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with Fetcher() as fetcher:
        for i, tid in enumerate(to_fetch, 1):
            try:
                html = fetcher.get_html(f"/team/{tid}")
                team = parse_team(html, team_id=tid)
                cache[str(tid)] = team.tag or team.name
                summary["fetched"] += 1
            except Exception as e:  # noqa: BLE001 -- keep going, one bad team page shouldn't kill the batch
                summary["failed"] += 1
                if verbose:
                    print(f"  failed team {tid}: {e}", file=sys.stderr)
            if verbose and i % 25 == 0:
                print(f"  [{i}/{len(to_fetch)}] team tags fetched", file=sys.stderr)
    TEAM_TAGS_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return summary


def _resolve_veto_team(tag: str, team1_id: int | None, team2_id: int | None, tag_cache: dict[str, str]) -> int | None:
    """Map a veto note's team label (e.g. "SEN") to team1_id or team2_id using
    the authoritative tag cache. Returns None (skip this match) if either
    team's tag is unknown or the label doesn't match either -- guessing would
    corrupt the win-rate table."""
    if not tag:
        return None
    tag_u = tag.strip().upper()
    t1_tag = tag_cache.get(str(team1_id), "").upper() if team1_id is not None else ""
    t2_tag = tag_cache.get(str(team2_id), "").upper() if team2_id is not None else ""
    if tag_u == t1_tag and tag_u != t2_tag:
        return team1_id
    if tag_u == t2_tag and tag_u != t1_tag:
        return team2_id
    return None


def test_naive_ban_rule(recent_n: int = 200, min_history_games: int = 3) -> dict:
    """Walk-forward test: does 'ban your lowest-winrate available map' predict
    each team's first ban? Win rates are computed only from matches strictly
    earlier (by match_id) than the one being tested -- no lookahead.

    A team needs at least `min_history_games` prior maps played (in total,
    across all maps) before it's included, to avoid the cold-start problem of
    scoring a team with zero history as either right or wrong by default.
    """
    if not VETO_DATASET_PATH.exists():
        raise FileNotFoundError(f"{VETO_DATASET_PATH} does not exist -- run collect_veto_dataset() first")
    if not TEAM_TAGS_PATH.exists():
        raise FileNotFoundError(f"{TEAM_TAGS_PATH} does not exist -- run build_team_tag_cache() first")

    tag_cache = json.loads(TEAM_TAGS_PATH.read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in VETO_DATASET_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows.sort(key=lambda r: r["match_id"])  # match_id order == chronological proxy

    # team_id -> map -> [wins, losses]; team_id -> full name (for map-result lookups)
    record: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    games_played: dict[int, int] = defaultdict(int)
    name_of_id: dict[int, str] = {}

    evaluations: list[dict] = []
    skipped_unresolved = 0

    for row in rows:
        t1_id, t2_id = row.get("team1_id"), row.get("team2_id")
        if t1_id is not None:
            name_of_id[t1_id] = row["team1"]
        if t2_id is not None:
            name_of_id[t2_id] = row["team2"]

        veto = [VetoStep(**s) for s in row["veto"]]
        first_ban = next((s for s in veto if s.action == "ban"), None)
        pool = sorted({s.map for s in veto})
        team_id = _resolve_veto_team(first_ban.team, t1_id, t2_id, tag_cache) if first_ban else None
        if first_ban and team_id is None:
            skipped_unresolved += 1

        if team_id is not None and len(pool) >= 3 and games_played[team_id] >= min_history_games:
            win_rates = {}
            for mp in pool:
                w, l = record[team_id][mp]
                win_rates[mp] = (w / (w + l)) if (w + l) > 0 else 0.5  # neutral prior if never played
            predicted_ban = min(win_rates, key=win_rates.get)
            evaluations.append(
                {
                    "match_id": row["match_id"],
                    "team": name_of_id.get(team_id, team_id),
                    "predicted_ban": predicted_ban,
                    "actual_ban": first_ban.map,
                    "correct": predicted_ban == first_ban.map,
                    "pool_size": len(pool),
                    "team_games_played": games_played[team_id],
                }
            )

        # Update history with this match's actual results (after evaluating on it)
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
            games_played[winner_id] += 1
            games_played[loser_id] += 1

    used = evaluations[-recent_n:]
    n_correct = sum(1 for e in used if e["correct"])
    return {
        "n": len(used),
        "n_correct": n_correct,
        "accuracy": (n_correct / len(used)) if used else None,
        "total_eligible_ever": len(evaluations),
        "skipped_unresolved_team_name": skipped_unresolved,
        "evaluations": used,
    }


def test_ban_rules(recent_n: int = 200, min_history_games: int = 3) -> dict:
    """Walk-forward test of every ban in the veto (not just the first) against
    two competing rules:

      * "own_worst"     -- ban the available map you personally have the
                            lowest win rate on
      * "deny_opp_best" -- ban the available map your opponent has the
                            highest win rate on (deny their strength, rather
                            than avoid your own weakness)

    Results are broken out per ban position (1st ban, 2nd ban, ...) since the
    two rules could plausibly matter differently early vs late in the veto,
    and pool size (hence the random-guess baseline) shrinks as picks/bans
    happen. Same walk-forward win-rate history and team-tag resolution as
    test_naive_ban_rule.
    """
    if not VETO_DATASET_PATH.exists():
        raise FileNotFoundError(f"{VETO_DATASET_PATH} does not exist -- run collect_veto_dataset() first")
    if not TEAM_TAGS_PATH.exists():
        raise FileNotFoundError(f"{TEAM_TAGS_PATH} does not exist -- run build_team_tag_cache() first")

    tag_cache = json.loads(TEAM_TAGS_PATH.read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in VETO_DATASET_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows.sort(key=lambda r: r["match_id"])

    record: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    games_played: dict[int, int] = defaultdict(int)

    def win_rate(team_id: int, mp: str) -> float:
        w, l = record[team_id][mp]
        return (w / (w + l)) if (w + l) > 0 else 0.5

    evaluations: list[dict] = []

    for row in rows:
        t1_id, t2_id = row.get("team1_id"), row.get("team2_id")
        veto = [VetoStep(**s) for s in row["veto"]]
        pool = sorted({s.map for s in veto})
        available = set(pool)
        ban_index = 0

        for step in veto:
            if step.action != "ban":
                if step.action in ("pick", "decider"):
                    available.discard(step.map)
                continue
            ban_index += 1
            team_id = _resolve_veto_team(step.team, t1_id, t2_id, tag_cache)
            if team_id is None or len(available) < 2:
                available.discard(step.map)
                continue
            opp_id = t2_id if team_id == t1_id else t1_id
            if opp_id is None or games_played[team_id] < min_history_games or games_played[opp_id] < min_history_games:
                available.discard(step.map)
                continue

            own_rates = {m: win_rate(team_id, m) for m in available}
            opp_rates = {m: win_rate(opp_id, m) for m in available}
            # Sort candidates before min/max so ties resolve deterministically
            # (alphabetically-first map) instead of depending on Python's
            # per-process string-hash randomization, which otherwise makes
            # results silently unreproducible whenever rates tie -- common
            # here since unseen team-map pairs all default to the same 0.5.
            min_own = min(own_rates.values())
            max_opp = max(opp_rates.values())
            own_worst_tied = sum(1 for v in own_rates.values() if v == min_own) > 1
            deny_opp_best_tied = sum(1 for v in opp_rates.values() if v == max_opp) > 1
            pred_own_worst = min(sorted(own_rates), key=own_rates.get)
            pred_deny_opp_best = max(sorted(opp_rates), key=opp_rates.get)

            evaluations.append(
                {
                    "match_id": row["match_id"],
                    "ban_index": ban_index,
                    "pool_size_at_ban": len(available),
                    "actual_ban": step.map,
                    "own_worst_correct": pred_own_worst == step.map,
                    "own_worst_tied": own_worst_tied,
                    "deny_opp_best_correct": pred_deny_opp_best == step.map,
                    "deny_opp_best_tied": deny_opp_best_tied,
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
            games_played[winner_id] += 1
            games_played[loser_id] += 1

    used = evaluations[-recent_n:]
    by_position: dict[int, dict] = {}
    for e in used:
        pos = e["ban_index"]
        bucket = by_position.setdefault(
            pos,
            {
                "n": 0, "own_worst_correct": 0, "deny_opp_best_correct": 0, "pool_sizes": [],
                "own_worst_clean_n": 0, "own_worst_clean_correct": 0,
                "deny_opp_best_clean_n": 0, "deny_opp_best_clean_correct": 0,
            },
        )
        bucket["n"] += 1
        bucket["own_worst_correct"] += e["own_worst_correct"]
        bucket["deny_opp_best_correct"] += e["deny_opp_best_correct"]
        bucket["pool_sizes"].append(e["pool_size_at_ban"])
        if not e["own_worst_tied"]:
            bucket["own_worst_clean_n"] += 1
            bucket["own_worst_clean_correct"] += e["own_worst_correct"]
        if not e["deny_opp_best_tied"]:
            bucket["deny_opp_best_clean_n"] += 1
            bucket["deny_opp_best_clean_correct"] += e["deny_opp_best_correct"]

    position_summary = {}
    for pos, b in sorted(by_position.items()):
        avg_pool = sum(b["pool_sizes"]) / len(b["pool_sizes"])
        position_summary[pos] = {
            "n": b["n"],
            "own_worst_accuracy": b["own_worst_correct"] / b["n"],
            "deny_opp_best_accuracy": b["deny_opp_best_correct"] / b["n"],
            "random_baseline": 1 / avg_pool,
            "avg_pool_size": avg_pool,
            # "clean" = excludes evaluations where the rule's prediction was a
            # tie broken arbitrarily (alphabetically) -- a purer read on
            # whether the rule has real predictive content.
            "own_worst_clean_n": b["own_worst_clean_n"],
            "own_worst_clean_accuracy": (b["own_worst_clean_correct"] / b["own_worst_clean_n"]) if b["own_worst_clean_n"] else None,
            "deny_opp_best_clean_n": b["deny_opp_best_clean_n"],
            "deny_opp_best_clean_accuracy": (b["deny_opp_best_clean_correct"] / b["deny_opp_best_clean_n"]) if b["deny_opp_best_clean_n"] else None,
        }

    return {
        "n": len(used),
        "total_eligible_ever": len(evaluations),
        "by_ban_position": position_summary,
    }
