"""Closing-line capture pipeline for vlr.gg's Betting widget.

vlr.gg only shows *paired* team1/team2 odds (needed to compute vig) while a
match is upcoming or live. The moment a match finishes, the widget collapses
to a single "$100 -> $payout" line for the winning side only, so a closing
line can only be captured by polling *before* the match ends -- there is no
way to retroactively reconstruct it for a match that already finished.

Design: each poll cycle
  1. fetches the current upcoming/live match list and snapshots pre-match/live
     odds for each one, overwriting a small per-match JSON state file each time
     (so the state file always holds the *latest* known snapshot);
  2. fetches the recent results list and, for any match that has now
     completed, "finalizes" it -- the last snapshot in its state file becomes
     the closing line, gets vig/implied-prob computed, and is appended to the
     long-format JSONL store. Matches that complete between polls without
     ever being snapshotted are recorded too, flagged `missed_pre_match=True`,
     with vig left blank since only the winner's payout is left on the page.

`export_excel` reads the JSONL store and writes a workbook with the three
known bookmakers (thunderpick/rainbet/shuffle) side by side, plus a long-format
sheet, so it can just be re-run after any poll to refresh the .xlsx.
"""

from __future__ import annotations

import difflib
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .fetch import Fetcher
from .models import BetLine, MatchSummary
from .parse import parse_match_odds, parse_matches

STATE_DIR = Path("data/closing_lines/state")
JSONL_PATH = Path("data/closing_lines/closing_lines.jsonl")
FINALIZED_IDS_PATH = Path("data/closing_lines/finalized_ids.json")
XLSX_PATH = Path("data/closing_lines/valorant_closing_lines.xlsx")
VIG_SAMPLES_PATH = Path("data/closing_lines/vig_samples.jsonl")
VIG_ESTIMATE_SAMPLE_SIZE = 10

KNOWN_BOOKMAKERS = ["thunderpick", "rainbet", "shuffle"]


@dataclass
class MatchState:
    match_id: int
    match_url: str
    date_display: str | None
    time_display: str | None
    team1: str | None
    team2: str | None
    event_name: str | None
    series: str | None
    captured_at: str
    lines: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_path(match_id: int) -> Path:
    return STATE_DIR / f"{match_id}.json"


def _save_state(state: MatchState) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _state_path(state.match_id).write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")


def _load_state(match_id: int) -> MatchState | None:
    path = _state_path(match_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return MatchState(**data)


def _load_finalized_ids() -> set[int]:
    if not FINALIZED_IDS_PATH.exists():
        return set()
    return set(json.loads(FINALIZED_IDS_PATH.read_text(encoding="utf-8")))


def _save_finalized_ids(ids: set[int]) -> None:
    FINALIZED_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINALIZED_IDS_PATH.write_text(json.dumps(sorted(ids)), encoding="utf-8")


def _classify_side(team: str | None, team_tag: str | None, team1_name: str | None, team2_name: str | None) -> str | None:
    """Best-effort match of a BetLine's team label to the match's team1/team2."""
    candidates = [c for c in (team, team_tag) if c]
    best_side, best_score = None, 0.0
    for c in candidates:
        cu = c.upper()
        for side, name in (("team1", team1_name), ("team2", team2_name)):
            if not name:
                continue
            nu = name.upper()
            score = difflib.SequenceMatcher(None, cu, nu).ratio()
            if cu in nu or nu.startswith(cu):
                score = max(score, 0.9)
            if score > best_score:
                best_score, best_side = score, side
    return best_side


def _pair_by_bookmaker(lines: list[BetLine], team1_name: str | None, team2_name: str | None) -> dict[str, dict]:
    """Group pre-match/live BetLines by bookmaker, keeping only fully-paired (both sides) books."""
    by_bookmaker: dict[str, list[BetLine]] = {}
    for line in lines:
        if line.stage == "post-match" or not line.bookmaker:
            continue
        by_bookmaker.setdefault(line.bookmaker, []).append(line)

    paired: dict[str, dict] = {}
    for bookmaker, bm_lines in by_bookmaker.items():
        if len(bm_lines) != 2:
            continue
        a, b = bm_lines
        side_a = _classify_side(a.team, a.team_tag, team1_name, team2_name)
        side_b = _classify_side(b.team, b.team_tag, team1_name, team2_name)
        if side_a == "team1" and side_b == "team2":
            team1_line, team2_line = a, b
        elif side_a == "team2" and side_b == "team1":
            team1_line, team2_line = b, a
        else:
            team1_line, team2_line = a, b  # fall back to widget order
        if not team1_line.decimal_odds or not team2_line.decimal_odds:
            continue
        paired[bookmaker] = {"team1_odds": team1_line.decimal_odds, "team2_odds": team2_line.decimal_odds}
    return paired


def _append_vig_sample(bookmaker: str, vig: float, sampled_at: str, source: str) -> None:
    """Log one observed (real, paired-odds) vig reading for later averaging.

    `source` is "closing" (captured right before a match went live/finished --
    the most representative) or "pre_match_snapshot" (an earlier snapshot of a
    still-upcoming match, used only when we don't yet have enough closing
    samples). Used to estimate vig for matches where we only ever saw the
    winner's one-sided payout odds.
    """
    VIG_SAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with VIG_SAMPLES_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"bookmaker": bookmaker, "vig": vig, "sampled_at": sampled_at, "source": source}) + "\n")


def estimate_vig_by_bookmaker(sample_size: int = VIG_ESTIMATE_SAMPLE_SIZE) -> dict[str, dict]:
    """Per-bookmaker vig estimate from the most recent real samples, preferring
    true closing-line samples over earlier pre-match snapshots when enough exist.

    Returns {bookmaker: {"mean": ..., "stddev": ..., "n": ..., "source": "closing"|"pre_match_snapshot"}}.
    stddev is the spread of the underlying samples -- a rough confidence signal
    for the estimate itself (tight spread = consistently-priced margin, wide
    spread = noisier estimate, often because the sample is really just a
    handful of matches resampled repeatedly rather than independent readings).
    """
    if not VIG_SAMPLES_PATH.exists():
        return {}
    rows = [json.loads(line) for line in VIG_SAMPLES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    estimates: dict[str, dict] = {}
    for bookmaker in {r["bookmaker"] for r in rows}:
        bm_rows = [r for r in rows if r["bookmaker"] == bookmaker]
        closing = [r["vig"] for r in bm_rows if r["source"] == "closing"][-sample_size:]
        used_closing = len(closing) >= min(3, sample_size)
        sample = closing if used_closing else [r["vig"] for r in bm_rows][-sample_size:]
        if sample:
            estimates[bookmaker] = {
                "mean": sum(sample) / len(sample),
                "stddev": statistics.pstdev(sample) if len(sample) > 1 else 0.0,
                "n": len(sample),
                "source": "closing" if used_closing else "pre_match_snapshot",
            }
    return estimates


def poll_once(upcoming_pages: int = 1, results_pages: int = 2, verbose: bool = True) -> dict:
    """One poll cycle: snapshot upcoming/live odds, finalize newly-completed matches.

    Returns a summary dict: {"snapshotted": N, "finalized": N, "missed": N}.
    """
    summary = {"snapshotted": 0, "finalized": 0, "missed": 0}
    finalized_ids = _load_finalized_ids()

    with Fetcher() as fetcher:
        upcoming: list[MatchSummary] = []
        for page in range(1, upcoming_pages + 1):
            html = fetcher.get_html("/matches", params={"page": page})
            upcoming.extend(parse_matches(html, source="global:upcoming"))

        pending = [m for m in upcoming if m.status in ("Upcoming", "LIVE")]
        for m in pending:
            odds_html = fetcher.get_html(m.match_url)
            lines = parse_match_odds(odds_html, match_id=m.match_id, source=f"match:{m.match_id}")
            active_lines = [l for l in lines if l.stage != "post-match"]
            if not active_lines:
                continue
            state = MatchState(
                match_id=m.match_id,
                match_url=m.match_url,
                date_display=m.date_display,
                time_display=m.time_display,
                team1=m.team1,
                team2=m.team2,
                event_name=m.event_name,
                series=m.series,
                captured_at=_now_iso(),
                lines=[l.to_dict() for l in active_lines],
            )
            _save_state(state)
            summary["snapshotted"] += 1
            if verbose:
                print(f"  snapshot match {m.match_id} ({m.team1} vs {m.team2}): {len(active_lines)} lines", file=sys.stderr)

            for bookmaker, odds in _pair_by_bookmaker(active_lines, m.team1, m.team2).items():
                sample_vig = 1 / odds["team1_odds"] + 1 / odds["team2_odds"] - 1
                _append_vig_sample(bookmaker, sample_vig, state.captured_at, source="pre_match_snapshot")

        results: list[MatchSummary] = []
        for page in range(1, results_pages + 1):
            html = fetcher.get_html("/matches/results", params={"page": page})
            results.extend(parse_matches(html, source="global:results"))

        with JSONL_PATH.open("a", encoding="utf-8") as jsonl_f:
            JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
            for m in results:
                if m.status != "Completed" or m.match_id in finalized_ids:
                    continue
                winner = m.team1 if m.team1_won else (m.team2 if m.team2_won else None)
                state = _load_state(m.match_id)

                if state is not None:
                    paired = _pair_by_bookmaker(
                        [BetLine(**d) for d in state.lines], state.team1, state.team2
                    )
                    for bookmaker, odds in paired.items():
                        implied1 = 1 / odds["team1_odds"]
                        implied2 = 1 / odds["team2_odds"]
                        vig = implied1 + implied2 - 1
                        row = {
                            "match_id": m.match_id,
                            "match_url": state.match_url,
                            "date_display": state.date_display,
                            "time_display": state.time_display,
                            "event_name": state.event_name,
                            "series": state.series,
                            "team1": state.team1,
                            "team2": state.team2,
                            "team1_score": m.team1_score,
                            "team2_score": m.team2_score,
                            "winner": winner,
                            "bookmaker": bookmaker,
                            "team1_odds": odds["team1_odds"],
                            "team2_odds": odds["team2_odds"],
                            "implied_prob_team1": implied1,
                            "implied_prob_team2": implied2,
                            "vig": vig,
                            "fair_prob_team1": implied1 / (implied1 + implied2),
                            "fair_prob_team2": implied2 / (implied1 + implied2),
                            "closing_captured_at": state.captured_at,
                            "missed_pre_match": False,
                        }
                        jsonl_f.write(json.dumps(row) + "\n")
                        _append_vig_sample(bookmaker, vig, state.captured_at, source="closing")
                    if paired:
                        summary["finalized"] += 1
                    _state_path(m.match_id).unlink(missing_ok=True)
                else:
                    # Never snapshotted pre-match (e.g. first run after it already
                    # finished) -- vlr.gg only leaves the winner's payout odds on the
                    # page, so exact vig is gone for good. We still record the raw
                    # winner-side odds/implied prob (exact), and export_excel later
                    # applies an *estimated* vig (avg of recent real vig samples) to
                    # back out an estimated fair probability for these rows.
                    odds_html = fetcher.get_html(m.match_url)
                    lines = parse_match_odds(odds_html, match_id=m.match_id, source=f"match:{m.match_id}")
                    post_lines = [l for l in lines if l.stage == "post-match" and l.decimal_odds]
                    for l in post_lines:
                        row = {
                            "match_id": m.match_id,
                            "match_url": m.match_url,
                            "date_display": m.date_display,
                            "time_display": m.time_display,
                            "event_name": m.event_name,
                            "series": m.series,
                            "team1": m.team1,
                            "team2": m.team2,
                            "team1_score": m.team1_score,
                            "team2_score": m.team2_score,
                            "winner": winner,
                            "bookmaker": l.bookmaker,
                            "winner_team": l.team,
                            "winner_odds": l.decimal_odds,
                            "winner_implied_prob_raw": l.implied_prob_raw,
                            "team1_odds": None,
                            "team2_odds": None,
                            "implied_prob_team1": None,
                            "implied_prob_team2": None,
                            "vig": None,
                            "fair_prob_team1": None,
                            "fair_prob_team2": None,
                            "closing_captured_at": None,
                            "missed_pre_match": True,
                        }
                        jsonl_f.write(json.dumps(row) + "\n")
                    summary["missed"] += 1

                finalized_ids.add(m.match_id)

    _save_finalized_ids(finalized_ids)
    return summary


def backfill_historical(target_total: int = 300, max_pages: int = 80, verbose: bool = True) -> dict:
    """Expand historical coverage further back through /matches/results.

    These are always missed_pre_match=True rows (winner-only odds, estimated
    vig applied at export time) since the pipeline wasn't polling before the
    match happened -- there's no way to recover the losing side's odds for
    matches that finished before this pipeline started tracking them.
    """
    finalized_ids = _load_finalized_ids()
    summary = {"already_had": len(finalized_ids), "added": 0, "pages_scanned": 0}
    needed = target_total - len(finalized_ids)
    if needed <= 0:
        return summary

    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with Fetcher() as fetcher:
        with JSONL_PATH.open("a", encoding="utf-8") as jsonl_f:
            page = 1
            while summary["added"] < needed and page <= max_pages:
                html = fetcher.get_html("/matches/results", params={"page": page})
                matches = parse_matches(html, source="global:results")
                summary["pages_scanned"] += 1
                if not matches:
                    break
                for m in matches:
                    if summary["added"] >= needed:
                        break
                    if m.status != "Completed" or m.match_id in finalized_ids:
                        continue
                    winner = m.team1 if m.team1_won else (m.team2 if m.team2_won else None)
                    odds_html = fetcher.get_html(m.match_url)
                    lines = parse_match_odds(odds_html, match_id=m.match_id, source=f"match:{m.match_id}")
                    post_lines = [l for l in lines if l.stage == "post-match" and l.decimal_odds]
                    for l in post_lines:
                        row = {
                            "match_id": m.match_id,
                            "match_url": m.match_url,
                            "date_display": m.date_display,
                            "time_display": m.time_display,
                            "event_name": m.event_name,
                            "series": m.series,
                            "team1": m.team1,
                            "team2": m.team2,
                            "team1_score": m.team1_score,
                            "team2_score": m.team2_score,
                            "winner": winner,
                            "bookmaker": l.bookmaker,
                            "winner_team": l.team,
                            "winner_odds": l.decimal_odds,
                            "winner_implied_prob_raw": l.implied_prob_raw,
                            "team1_odds": None,
                            "team2_odds": None,
                            "implied_prob_team1": None,
                            "implied_prob_team2": None,
                            "vig": None,
                            "fair_prob_team1": None,
                            "fair_prob_team2": None,
                            "closing_captured_at": None,
                            "missed_pre_match": True,
                        }
                        jsonl_f.write(json.dumps(row) + "\n")
                    finalized_ids.add(m.match_id)
                    summary["added"] += 1
                    if verbose:
                        print(
                            f"  [{summary['added']}/{needed}] match {m.match_id} "
                            f"({m.team1} vs {m.team2}): {len(post_lines)} bookmakers",
                            file=sys.stderr,
                        )
                page += 1
    _save_finalized_ids(finalized_ids)
    return summary


def export_excel(jsonl_path: Path = JSONL_PATH, xlsx_path: Path = XLSX_PATH) -> int:
    """Rebuild the Excel workbook (long + wide-by-bookmaker sheets) from the JSONL store."""
    import pandas as pd

    if not jsonl_path.exists():
        raise FileNotFoundError(f"{jsonl_path} does not exist yet -- run `closing poll` first")

    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    vig_estimates = estimate_vig_by_bookmaker()
    for row in rows:
        if not row.get("missed_pre_match"):
            # Real paired odds: derive a uniform "winner side" view for consistency
            # with missed rows, purely for convenience -- vig/fair_prob here are exact.
            if row.get("winner") == row.get("team1"):
                row["winner_odds"] = row.get("team1_odds")
                row["winner_implied_prob_raw"] = row.get("implied_prob_team1")
                row["winner_fair_prob"] = row.get("fair_prob_team1")
            elif row.get("winner") == row.get("team2"):
                row["winner_odds"] = row.get("team2_odds")
                row["winner_implied_prob_raw"] = row.get("implied_prob_team2")
                row["winner_fair_prob"] = row.get("fair_prob_team2")
            row["estimated_vig"] = row.get("vig")
            row["estimated_vig_stddev"] = None
            row["estimated_vig_n"] = None
            row["vig_is_estimated"] = False
        else:
            # Historical match where only the winner's payout odds survived on
            # vlr.gg -- back out an estimated fair probability using the
            # bookmaker's recent average real vig (see estimate_vig_by_bookmaker).
            est = vig_estimates.get(row.get("bookmaker"))
            est_vig = est["mean"] if est else None
            p1 = row.get("winner_implied_prob_raw")
            row["estimated_vig"] = est_vig
            row["estimated_vig_stddev"] = est["stddev"] if est else None
            row["estimated_vig_n"] = est["n"] if est else None
            row["vig_is_estimated"] = True
            if est_vig is not None and p1 is not None:
                p2_est = max(0.0, 1 + est_vig - p1)
                row["winner_fair_prob"] = p1 / (p1 + p2_est) if (p1 + p2_est) > 0 else None
            else:
                row["winner_fair_prob"] = None

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        raise ValueError(f"{jsonl_path} has no rows yet")

    # A (match_id, bookmaker) pair can rarely end up double-written -- e.g. the
    # scheduled poll finalizes a match with real paired odds at the same time a
    # one-off backfill/recheck script independently records it as missed. Prefer
    # the real (non-missed, exact-vig) row when both exist for the same pair.
    long_df = long_df.sort_values("missed_pre_match").drop_duplicates(subset=["match_id", "bookmaker"], keep="first")

    match_cols = [
        "match_id", "date_display", "time_display", "event_name", "series",
        "team1", "team2", "team1_score", "team2_score", "winner",
    ]
    wide = long_df.drop_duplicates("match_id")[match_cols].set_index("match_id")

    bookmakers = list(dict.fromkeys(KNOWN_BOOKMAKERS + sorted(long_df["bookmaker"].dropna().unique())))
    metric_cols = [
        "team1_odds", "team2_odds", "implied_prob_team1", "implied_prob_team2",
        "vig", "fair_prob_team1", "fair_prob_team2", "closing_captured_at", "missed_pre_match",
        "winner_odds", "winner_implied_prob_raw", "winner_fair_prob",
        "estimated_vig", "estimated_vig_stddev", "estimated_vig_n", "vig_is_estimated",
    ]
    for bm in bookmakers:
        bm_rows = long_df[long_df["bookmaker"] == bm].set_index("match_id")
        for col in metric_cols:
            wide[f"{bm}_{col}"] = bm_rows[col] if col in bm_rows.columns else None

    vig_cols = [f"{bm}_vig" for bm in bookmakers if f"{bm}_vig" in wide.columns]
    wide["avg_vig"] = wide[vig_cols].mean(axis=1, skipna=True) if vig_cols else None

    # Cross-bookmaker disagreement per match: spread of each book's *fair* (de-vigged)
    # win probability for the same outcome. High spread = the books are pricing this
    # match very differently; needs 2+ books with a fair-prob reading to be meaningful.
    fair_cols = [f"{bm}_winner_fair_prob" for bm in bookmakers if f"{bm}_winner_fair_prob" in wide.columns]

    def _disagreement(row):
        vals = [row[c] for c in fair_cols if pd.notna(row[c])]
        return statistics.pstdev(vals) if len(vals) >= 2 else None

    wide["book_disagreement_stddev"] = wide.apply(_disagreement, axis=1) if fair_cols else None
    wide["book_disagreement_n"] = wide.apply(lambda row: sum(1 for c in fair_cols if pd.notna(row[c])), axis=1) if fair_cols else 0

    wide = wide.reset_index().sort_values("match_id", ascending=False)
    long_df = long_df.sort_values(["match_id", "bookmaker"], ascending=[False, True])

    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        wide.to_excel(writer, sheet_name="Closing Lines (wide)", index=False)
        long_df.to_excel(writer, sheet_name="Closing Lines (long)", index=False)

    return len(wide)
