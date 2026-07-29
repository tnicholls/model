"""VLR.gg's own team rating, scraped from the "Form Rating" chart on each
team's page (`/team/{id}/{slug}`), replacing our own from-scratch Elo as the
base-model feature. Rationale: our Elo only has 12 months / 46 teams to
learn from (K landed at the top of its grid -- a rating starved of updates
compensates by moving fast); VLR's is computed over their full multi-year,
all-tier database, so it should be far better converged.

VLR doesn't publish the formula (confirmed via their own forums: they
declined to disclose criteria/weightings). What's known only from user
reports: Elo-like, opponent-quality weighted, recency-weighted but favoring
long-term consistency, unknown whether round differential or international-
match weighting feeds in. Treat it as a better-converged but structurally
opaque signal, not a documented one.

Two different "VLR ratings" exist -- team ranking points (~2000-2300 scale,
what this module scrapes) and per-player performance rating (~1.15 scale,
parse.PlayerStat.rating). Confirmed which one the chart shows via its
y-axis values (1700-2050 range for Sentinels) and axis label text ("Rating"
on a "Matches Played" x-axis) -- unambiguous, not the player-rating scale.

The chart is server-rendered inline SVG (not a canvas/JS-fetched chart):
axis gridlines carry the value calibration (text labels next to grid-line
y-positions -> a linear y_pixel-to-rating regression), and each data point
is a `circle.data-hover` whose `onclick` links to the specific match. A team
may have multiple "core" (roster) rating charts on one page (old rosters
get their own chart, found via a "Core ID" selector) -- all found charts'
points are collected and joined purely by match_id, sidestepping the need
to model the core-switching UI.

CRITICAL (verified, not assumed): each point is the rating AFTER that
match, not before. Checked by testing whether consecutive-point deltas
correlate with the win/loss of the earlier or the later match in our own
dataset -- 44/44 (100%) matched "delta explained by the later match's
result", 0-for-not against the other hypothesis. So the pre-match rating
for match i is the most recent prior point's value, strictly before i --
never the value attached to i itself.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from bs4 import BeautifulSoup

from . import agent_proficiency as ap
from .fetch import Fetcher

RAW_POINTS_PATH = Path("data/vlr_rating/raw_points.json")
PRE_MATCH_RATINGS_PATH = Path("data/vlr_rating/pre_match_ratings.json")

_MATCH_HREF_RE = re.compile(r"^/(\d+)/")


def _extract_rating_graph(svg: BeautifulSoup) -> list[tuple[int, float]] | None:
    """(match_id, rating_after_this_match) pairs from one `svg.graph`
    element, or None if this SVG is the "Ranking History" (position) chart
    rather than the "Form Rating" (points) chart -- distinguished by the
    y-axis label text, not by chart order (a team can have zero, one, or
    two-per-core "Rating" charts on its page)."""
    axis_labels = [t.get_text(strip=True) for al in svg.find_all("g", class_="axis-label") for t in al.find_all("text")]
    if "Rating" not in axis_labels:
        return None
    labels_group = svg.find("g", class_="labels")
    if labels_group is None:
        return None
    calib = [(float(t.get("y")), float(t.get_text(strip=True))) for t in labels_group.find_all("text")]
    if len(calib) < 2:
        return None
    n = len(calib)
    mean_y = sum(c[0] for c in calib) / n
    mean_v = sum(c[1] for c in calib) / n
    num = sum((c[0] - mean_y) * (c[1] - mean_v) for c in calib)
    den = sum((c[0] - mean_y) ** 2 for c in calib)
    if den == 0:
        return None
    slope = num / den
    intercept = mean_v - slope * mean_y

    points: list[tuple[int, float]] = []
    for c in svg.find_all("circle", class_="data-hover"):
        onclick = c.get("onclick", "")
        href_match = re.search(r"href='([^']+)'", onclick)
        if not href_match:
            continue
        mid_match = _MATCH_HREF_RE.match(href_match.group(1))
        if not mid_match:
            continue
        cy = float(c.get("cy"))
        points.append((int(mid_match.group(1)), slope * cy + intercept))
    return points


def parse_team_rating_points(html: str) -> list[tuple[int, float]]:
    """All (match_id, rating_after) points from every "Form Rating" chart
    found on a team page (merged across roster cores, deduped by match_id
    keeping the first occurrence)."""
    soup = BeautifulSoup(html, "lxml")
    seen: dict[int, float] = {}
    for svg in soup.find_all("svg", class_="graph"):
        pts = _extract_rating_graph(svg)
        if not pts:
            continue
        for mid, val in pts:
            seen.setdefault(mid, val)
    return list(seen.items())


def collect_vlr_ratings(team_ids: list[int] | None = None, verbose: bool = True) -> dict:
    """Fetch every T1 team's page and extract its Form Rating chart
    point(s). Raw (match_id, rating_after) pairs saved per team_id --
    walk-forward conversion to a pre-match feature happens separately in
    build_pre_match_ratings, against OUR OWN dataset's true match dates."""
    if team_ids is None:
        teams_by_region = json.loads(Path("data/t1/teams.json").read_text(encoding="utf-8"))
        team_ids = sorted({int(tid) for region in teams_by_region.values() for tid in region})

    raw: dict[str, list[list]] = {}
    with Fetcher() as fetcher:
        for i, tid in enumerate(team_ids, 1):
            try:
                html = fetcher.get_html(f"/team/{tid}/team")
                pts = parse_team_rating_points(html)
                raw[str(tid)] = [[mid, val] for mid, val in pts]
                if verbose and i % 10 == 0:
                    print(f"  [{i}/{len(team_ids)}] teams fetched", file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                raw[str(tid)] = []
                if verbose:
                    print(f"  failed team {tid}: {e}", file=sys.stderr)

    RAW_POINTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_POINTS_PATH.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return {"n_teams": len(team_ids), "n_teams_with_points": sum(1 for v in raw.values() if v), "n_total_points": sum(len(v) for v in raw.values())}


def build_pre_match_ratings() -> dict:
    """For every (match_id, team_id) pair in OUR dataset, the VLR rating as
    of strictly before that match: the most recent scraped point (by our
    own true match date, not vlr.gg's chart pixel order) that is < this
    match's date, for that team. None if no such point exists (team has no
    scraped history before this match -- new/promoted team or scrape gap).
    """
    raw = json.loads(RAW_POINTS_PATH.read_text(encoding="utf-8"))
    date_by_match = ap.load_date_by_match()

    # team_id -> sorted [(date, rating_after), ...] using OUR dates, not chart pixel order
    team_series: dict[int, list[tuple[str, float]]] = defaultdict(list)
    n_points_undated = 0
    for tid_str, points in raw.items():
        tid = int(tid_str)
        for mid, val in points:
            d = date_by_match.get(mid)
            if d is None:
                n_points_undated += 1
                continue
            team_series[tid].append((d, val))
    for tid in team_series:
        team_series[tid].sort(key=lambda t: t[0])

    matches = ap.load_matches_ordered()
    pre_match: dict[str, dict] = {}
    for m in matches:
        match_id = m["match_id"]
        date = date_by_match.get(match_id)
        if date is None:
            continue
        entry = {}
        for team_id in (m.get("team1_id"), m.get("team2_id")):
            if team_id is None:
                continue
            series = team_series.get(team_id, [])
            prior = [v for d, v in series if d < date]
            entry[str(team_id)] = prior[-1] if prior else None
        pre_match[str(match_id)] = entry

    PRE_MATCH_RATINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PRE_MATCH_RATINGS_PATH.write_text(json.dumps(pre_match, indent=2), encoding="utf-8")
    return {"n_matches": len(pre_match), "n_undated_scraped_points_dropped": n_points_undated}


def coverage_report() -> dict:
    """How many of OUR matches have a pre-match VLR rating for BOTH teams,
    and which teams have the most gaps (missing pre-match rating) -- flags
    newly-promoted/formed teams with provisional or absent VLR history."""
    pre_match = json.loads(PRE_MATCH_RATINGS_PATH.read_text(encoding="utf-8"))
    matches = {m["match_id"]: m for m in ap.load_matches_ordered()}

    n_both = n_one = n_neither = 0
    gaps_by_team: dict[str, int] = defaultdict(int)
    total_by_team: dict[str, int] = defaultdict(int)

    for mid_str, entry in pre_match.items():
        m = matches.get(int(mid_str))
        if m is None:
            continue
        t1_id, t2_id = m.get("team1_id"), m.get("team2_id")
        if t1_id is None or t2_id is None:
            continue
        r1 = entry.get(str(t1_id))
        r2 = entry.get(str(t2_id))
        for tid, r in ((t1_id, r1), (t2_id, r2)):
            total_by_team[str(tid)] += 1
            if r is None:
                gaps_by_team[str(tid)] += 1
        if r1 is not None and r2 is not None:
            n_both += 1
        elif r1 is not None or r2 is not None:
            n_one += 1
        else:
            n_neither += 1

    worst_gaps = sorted(
        ((tid, gaps, total_by_team[tid]) for tid, gaps in gaps_by_team.items() if gaps > 0),
        key=lambda t: -t[1] / t[2],
    )[:15]

    return {
        "n_matches_total": len(pre_match),
        "n_both_teams_covered": n_both,
        "n_one_team_covered": n_one,
        "n_neither_covered": n_neither,
        "coverage_pct_both": n_both / len(pre_match) if pre_match else None,
        "worst_gap_teams": [{"team_id": tid, "missing": g, "total": t, "missing_pct": g / t} for tid, g, t in worst_gaps],
    }
