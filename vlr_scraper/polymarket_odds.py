"""Polymarket odds capture for T1 Valorant matches (Americas/EMEA/Pacific/China
+ the international events those franchises play in: Masters, Champions,
Esports World Cup, Kickoff, ...).

Why this exists / how it differs from closing_lines.py: that pipeline scrapes
vlr.gg's Betting widget (thunderpick/rainbet/shuffle), which only ever shows
the *current* odds -- there is no history endpoint, so capturing a closing
line requires polling in real time before the match finishes. Polymarket is
the opposite: its CLOB exposes a genuine historical price time series per
outcome token (`/prices-history`), so a match that already happened can be
backfilled after the fact -- no need to have been polling when it happened.
The one thing that *can't* be backfilled is order-book depth (bids/asks) at
a past moment -- only the live book is queryable -- so historical liquidity
is approximated from the historical trade tape instead (see
`trades_in_window`); genuine live order-book depth is only captured going
forward, by `poll_once`.

Market structure (confirmed against the live API, see module docstring below
for the reasoning): a Bo-`x` match event has exactly `x - 1` "Map N Winner"
child markets (N = 1..x-1) plus one parent "match winner" market. There is no
separate market for the deciding map -- once the first `x-1` maps are split
so neither side has reached the (x+1)//2 wins needed to close it out early,
the parent match-winner market's price *is* the deciding-map price, because
whoever wins that last map wins the match. `decider_anchor_snapshot` captures
that moment using the last pre-decider map market's `closedTime` (the instant
Polymarket resolved it) as the anchor.
"""

from __future__ import annotations

import difflib
import json
import re
import statistics
import sys
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .polymarket_fetch import PolymarketFetcher

T1_TEAMS_PATH = Path("data/t1/teams.json")
OUT_DIR = Path("data/polymarket_odds")
JSONL_PATH = OUT_DIR / "polymarket_odds.jsonl"
XLSX_PATH = OUT_DIR / "polymarket_valorant_odds.xlsx"
STATE_DIR = OUT_DIR / "state"
FINALIZED_IDS_PATH = OUT_DIR / "finalized_ids.json"

REGION_MARKERS = {
    "americas": ["vct americas", "vct 2025: americas", "vct 2026: americas"],
    "emea": ["vct emea", "vct 2025: emea", "vct 2026: emea"],
    "pacific": ["vct pacific", "vct 2025: pacific", "vct 2026: pacific"],
    "china": ["vct china", "vct 2025: china", "vct 2026: china"],
}
INTL_MARKERS = ["masters", "champions tour", "valorant champions", "esports world cup", "kickoff", "red bull home ground"]
# Explicit non-T1 divisions -- excluded even if a team name happens to
# substring-match a T1 org (e.g. "GIANTX GC" containing "GIANTX").
EXCLUDE_MARKERS = ["game changers", "challengers", "collegiate", " vcl", "academy", "last chance qualifier"]

TEAM_NAME_MATCH_THRESHOLD = 0.90

MAP_MARKET_RE = re.compile(r"^Map (\d+) Winner$", re.IGNORECASE)
BO_RE = re.compile(r"\(BO(\d)\)", re.IGNORECASE)

ANCHOR_OFFSETS = {
    "t_minus_24h": timedelta(hours=24),
    "t_minus_12h": timedelta(hours=12),
    "pre_match": timedelta(seconds=0),
}
TRADES_WINDOW_SECONDS = 3600  # +/- 1h around a snapshot for the liquidity proxy


@dataclass
class PMMarket:
    condition_id: str
    question: str
    map_num: int | None  # None for the parent match-winner market
    outcomes: list[str]
    clob_token_ids: list[str]
    game_start_time: str | None
    closed_time: str | None
    outcome_prices: list[float] | None
    closed: bool

    def winner_index(self) -> int | None:
        if not self.outcome_prices:
            return None
        best_i, best_p = None, -1.0
        for i, p in enumerate(self.outcome_prices):
            if p > best_p:
                best_i, best_p = i, p
        return best_i if best_p >= 0.5 else None


@dataclass
class PMMatch:
    event_id: str
    slug: str
    title: str
    region: str  # americas/emea/pacific/china/international/other
    bo_x: int
    team_a: str
    team_b: str
    team_a_t1_name: str | None
    team_b_t1_name: str | None
    team_a_score: float
    team_b_score: float
    game_start_time: str | None  # ISO
    match_winner: PMMarket
    map_markets: list[PMMarket]  # ordered by map_num


def _clean_t1_name(raw: str) -> str:
    return raw.split("\t")[0].split("\n")[0].strip()


def load_t1_roster() -> list[dict]:
    """[{"region":..., "team_id":..., "name":...}] from data/t1/teams.json."""
    data = json.loads(T1_TEAMS_PATH.read_text(encoding="utf-8"))
    roster = []
    for region, teams in data.items():
        for team_id, raw_name in teams.items():
            roster.append({"region": region, "team_id": int(team_id), "name": _clean_t1_name(raw_name)})
    return roster


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^A-Za-z0-9 ]", " ", s).upper()
    return re.sub(r"\s+", " ", s).strip()


def match_team_name(candidate: str, roster: list[dict]) -> dict | None:
    """Best T1-roster match for a Polymarket outcome name, or None if nothing
    clears TEAM_NAME_MATCH_THRESHOLD. Exact normalized match short-circuits;
    otherwise a strict similarity ratio -- deliberately strict, because loose
    matching would let "GIANTX GC" (Game Changers) match "GIANTX" (T1)."""
    nc = _normalize(candidate)
    best, best_score = None, 0.0
    for t in roster:
        nt = _normalize(t["name"])
        if nc == nt:
            return {**t, "score": 1.0}
        score = difflib.SequenceMatcher(None, nc, nt).ratio()
        if score > best_score:
            best, best_score = t, score
    if best is not None and best_score >= TEAM_NAME_MATCH_THRESHOLD:
        return {**best, "score": best_score}
    return None


def classify_region(title: str) -> str:
    tl = title.lower()
    for region, markers in REGION_MARKERS.items():
        if any(m in tl for m in markers):
            return region
    if any(m in tl for m in INTL_MARKERS):
        return "international"
    return "other"


def _iso_to_ts(iso: str | None) -> int | None:
    if not iso:
        return None
    s = iso.replace("Z", "+00:00")
    try:
        return int(datetime.fromisoformat(s).timestamp())
    except ValueError:
        return None


def parse_pm_match(event: dict, roster: list[dict]) -> PMMatch | None:
    """Extract a T1-vs-T1 match (parent market + its map-winner children) from
    a raw Gamma event, or None if this event isn't a T1 head-to-head match
    (outright/handicap/totals-only events, non-T1 divisions, no team match)."""
    title = event.get("title", "")
    tl = title.lower()
    if " vs " not in tl and " vs. " not in tl:
        return None
    if any(m in tl for m in EXCLUDE_MARKERS):
        return None

    parent = None
    map_markets: dict[int, PMMarket] = {}
    for m in event.get("markets", []):
        group_title = (m.get("groupItemTitle") or "").strip()
        map_match = MAP_MARKET_RE.match(group_title)
        outcomes = m.get("outcomes") or []
        token_ids = m.get("clobTokenIds") or []
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if isinstance(token_ids, str):
            token_ids = json.loads(token_ids)
        if len(outcomes) != 2 or len(token_ids) != 2:
            continue
        outcome_prices = m.get("outcomePrices")
        if isinstance(outcome_prices, str):
            outcome_prices = json.loads(outcome_prices)
        pm = PMMarket(
            condition_id=m.get("conditionId"),
            question=m.get("question", ""),
            map_num=int(map_match.group(1)) if map_match else None,
            outcomes=outcomes,
            clob_token_ids=token_ids,
            game_start_time=m.get("gameStartTime") or m.get("eventStartTime"),
            closed_time=m.get("closedTime"),
            outcome_prices=[float(p) for p in outcome_prices] if outcome_prices else None,
            closed=bool(m.get("closed")),
        )
        if map_match:
            map_markets[pm.map_num] = pm
        elif group_title in ("", "Game Winner", "Match Winner"):
            parent = pm

    if parent is None:
        return None

    team_a_raw, team_b_raw = parent.outcomes
    match_a = match_team_name(team_a_raw, roster)
    match_b = match_team_name(team_b_raw, roster)
    if not match_a or not match_b or match_a["team_id"] == match_b["team_id"]:
        return None

    bo_match = BO_RE.search(title)
    ordered_maps = [map_markets[k] for k in sorted(map_markets)]
    bo_x = int(bo_match.group(1)) if bo_match else (max(map_markets) + 1 if map_markets else 3)

    # Title text is the primary signal (it's explicit about which bracket,
    # e.g. "VCT Pacific Group Omega" vs "Masters London"), but plenty of
    # regional-stage events don't name the region in the title at all (e.g.
    # "Valorant: Wolves Esports vs FunPlus Phoenix (BO3)" -- both China
    # teams, no region marker). Fall back to the T1 roster's own region for
    # those, and only fall further back to "other" when the two teams are
    # rostered in *different* regions without an international title marker
    # (shouldn't normally happen -- would mean a title-classification miss).
    region = classify_region(title)
    if region == "other":
        if match_a["region"] == match_b["region"]:
            region = match_a["region"]
        else:
            region = "international"

    return PMMatch(
        event_id=str(event.get("id")),
        slug=event.get("slug", ""),
        title=title,
        region=region,
        bo_x=bo_x,
        team_a=team_a_raw,
        team_b=team_b_raw,
        team_a_t1_name=match_a["name"],
        team_b_t1_name=match_b["name"],
        team_a_score=match_a["score"],
        team_b_score=match_b["score"],
        game_start_time=parent.game_start_time,
        match_winner=parent,
        map_markets=ordered_maps,
    )


def discover_events(fetcher: PolymarketFetcher, verbose: bool = True) -> list[dict]:
    """Every Valorant event on Polymarket, open + closed, paginated 100 at a time."""
    events: list[dict] = []
    for closed in (False, True):
        offset = 0
        while True:
            page = fetcher.get_gamma(
                "/events", params={"tag_slug": "valorant", "closed": str(closed).lower(), "limit": 100, "offset": offset}
            )
            if not page:
                break
            events.extend(page)
            if verbose:
                print(f"  discover: closed={closed} offset={offset} -> {len(page)} events", file=sys.stderr)
            if len(page) < 100:
                break
            offset += 100
    return events


def discover_t1_matches(fetcher: PolymarketFetcher, verbose: bool = True) -> list[PMMatch]:
    roster = load_t1_roster()
    events = discover_events(fetcher, verbose=verbose)
    matches = []
    for e in events:
        m = parse_pm_match(e, roster)
        if m is not None:
            matches.append(m)
    if verbose:
        by_region: dict[str, int] = {}
        for m in matches:
            by_region[m.region] = by_region.get(m.region, 0) + 1
        print(f"  matched {len(matches)}/{len(events)} events to T1 head-to-heads: {by_region}", file=sys.stderr)
    return matches


# ---------------------------------------------------------------------------
# Price history + vig + liquidity
# ---------------------------------------------------------------------------


def fetch_price_series(
    fetcher: PolymarketFetcher, token_id: str, start_ts: int | None = None, end_ts: int | None = None, fidelity: int = 60
) -> list[tuple[int, float]]:
    params: dict = {"market": token_id, "fidelity": fidelity}
    if start_ts is not None and end_ts is not None:
        params["startTs"] = start_ts
        params["endTs"] = end_ts
    else:
        params["interval"] = "max"
    data = fetcher.get_clob("/prices-history", params=params)
    history = (data or {}).get("history") or []
    return sorted((int(pt["t"]), float(pt["p"])) for pt in history)


def nearest_price(series: list[tuple[int, float]], target_ts: int, direction: str = "nearest") -> dict | None:
    """direction: 'nearest' (closest point either side), 'le' (last point at/before
    target, falling back to nearest if the series starts after target), or 'ge'
    (first point at/after target, falling back to nearest)."""
    if not series:
        return None
    if direction == "le":
        candidates = [p for p in series if p[0] <= target_ts]
        pick = max(candidates, key=lambda p: p[0]) if candidates else min(series, key=lambda p: abs(p[0] - target_ts))
    elif direction == "ge":
        candidates = [p for p in series if p[0] >= target_ts]
        pick = min(candidates, key=lambda p: p[0]) if candidates else min(series, key=lambda p: abs(p[0] - target_ts))
    else:
        pick = min(series, key=lambda p: abs(p[0] - target_ts))
    return {"ts": pick[0], "price": pick[1], "gap_seconds": pick[0] - target_ts}


def snapshot_pair(
    fetcher: PolymarketFetcher,
    market: PMMarket,
    target_ts: int,
    direction: str = "nearest",
    series_cache: dict | None = None,
) -> dict | None:
    """Vig-bearing snapshot of both sides of one market at one point in time.

    On a low-volume market, one outcome token can have an entirely empty
    prices-history series even though real trades exist for the market
    (observed: CLOB only indexes a price point for whichever token side
    actually traded). When that happens, derive the missing side from its
    complement's series (price_missing = 1 - price_other) rather than
    dropping the snapshot -- these are complementary CTF tokens, so that's
    an approximation, not a guess, though it can't reflect that side's own
    bid/ask spread."""
    cache = series_cache if series_cache is not None else {}
    for token_id in market.clob_token_ids:
        if token_id not in cache:
            cache[token_id] = fetch_price_series(fetcher, token_id)

    series_0, series_1 = cache[market.clob_token_ids[0]], cache[market.clob_token_ids[1]]
    price_source = "prices_history"
    if not series_0 and series_1:
        series_0 = [(t, 1 - p) for t, p in series_1]
        price_source = "complement_derived"
    if not series_1 and series_0:
        series_1 = [(t, 1 - p) for t, p in series_0]
        price_source = "complement_derived"

    if not series_0 and not series_1:
        price_source = "trades_fallback"
        # Observed on thin/low-volume markets: prices-history comes back
        # empty for *both* tokens even though real trades exist (the
        # aggregation service appears to skip building a series below some
        # activity threshold). Fall back to the trade tape itself -- the
        # executed price of the nearest real trade, per side.
        trades_key = f"trades:{market.condition_id}"
        if trades_key not in cache:
            cache[trades_key] = fetch_trades(fetcher, market.condition_id)
        trades = cache[trades_key]
        series_0 = sorted((t["timestamp"], t["price"]) for t in trades if t.get("outcomeIndex") == 0)
        series_1 = sorted((t["timestamp"], t["price"]) for t in trades if t.get("outcomeIndex") == 1)
        if not series_0 and series_1:
            series_0 = [(t, 1 - p) for t, p in series_1]
        if not series_1 and series_0:
            series_1 = [(t, 1 - p) for t, p in series_0]

    prices = [nearest_price(series_0, target_ts, direction=direction), nearest_price(series_1, target_ts, direction=direction)]
    if prices[0] is None or prices[1] is None:
        return None
    pa, pb = prices[0]["price"], prices[1]["price"]
    total = pa + pb
    return {
        "anchor_target_ts": target_ts,
        "price_team_a": pa,
        "price_team_b": pb,
        "vig": total - 1,
        "fair_prob_team_a": pa / total if total > 0 else None,
        "fair_prob_team_b": pb / total if total > 0 else None,
        "gap_seconds_a": prices[0]["gap_seconds"],
        "gap_seconds_b": prices[1]["gap_seconds"],
        "actual_ts_a": prices[0]["ts"],
        "actual_ts_b": prices[1]["ts"],
        "price_source": price_source,
    }


def fetch_trades(fetcher: PolymarketFetcher, condition_id: str, max_trades: int = 5000) -> list[dict]:
    trades: list[dict] = []
    offset = 0
    page_size = 500
    while len(trades) < max_trades:
        page = fetcher.get_data_api("/trades", params={"market": condition_id, "limit": page_size, "offset": offset})
        if not page:
            break
        trades.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
    return trades


def trades_in_window(trades: list[dict], center_ts: int, window_seconds: int = TRADES_WINDOW_SECONDS) -> dict:
    lo, hi = center_ts - window_seconds, center_ts + window_seconds
    in_window = [t for t in trades if lo <= t.get("timestamp", 0) <= hi]
    total_size = sum(t.get("size", 0.0) for t in in_window)
    notional = sum(t.get("size", 0.0) * t.get("price", 0.0) for t in in_window)
    return {"trades_n": len(in_window), "trades_size": total_size, "trades_notional": notional}


def decider_target(match: PMMatch) -> tuple[int, int] | None:
    """(anchor_ts, decider_map_num) if this match's map results are fully known
    and the series was forced to the deciding map, else None."""
    if not match.map_markets or len(match.map_markets) < match.bo_x - 1:
        return None
    winners = [mm.winner_index() for mm in match.map_markets]
    if any(w is None for w in winners):
        return None
    counts = [winners.count(0), winners.count(1)]
    needed = (match.bo_x + 1) // 2
    if max(counts) >= needed:
        return None  # closed out before the decider
    last_map = match.map_markets[-1]
    ts = _iso_to_ts(last_map.closed_time)
    if ts is None:
        return None
    return ts, match.bo_x


# ---------------------------------------------------------------------------
# Backfill (historical, price-history based -- no live polling required)
# ---------------------------------------------------------------------------


def _market_rows(fetcher: PolymarketFetcher, match: PMMatch, market_label: str, market: PMMarket) -> list[dict]:
    if match.game_start_time is None:
        return []
    start_ts = _iso_to_ts(match.game_start_time)
    if start_ts is None:
        return []
    series_cache: dict = {}
    rows = []
    for anchor_label, offset in ANCHOR_OFFSETS.items():
        target_ts = start_ts - int(offset.total_seconds())
        # "pre_match" must never leak a post-kickoff price (info from an
        # already-started map) into what's supposed to be the closing line,
        # so it's pinned to the last point at/before kickoff. The 24h/12h
        # anchors tolerate a point on either side -- nearest is fine there.
        direction = "le" if anchor_label == "pre_match" else "nearest"
        snap = snapshot_pair(fetcher, market, target_ts, direction=direction, series_cache=series_cache)
        if snap is None:
            continue
        rows.append({**_base_row(match), "market": market_label, "anchor": anchor_label, **snap})

    if market_label == "match_winner":
        decider = decider_target(match)
        if decider is not None:
            decider_ts, decider_map_num = decider
            snap = snapshot_pair(fetcher, market, decider_ts, direction="ge", series_cache=series_cache)
            if snap is not None:
                rows.append(
                    {
                        **_base_row(match),
                        "market": f"map{decider_map_num}_decider",
                        "anchor": "decider",
                        "decider_anchor_source": "previous_map_closed_time",
                        **snap,
                    }
                )
    return rows


def _base_row(match: PMMatch) -> dict:
    return {
        "event_id": match.event_id,
        "slug": match.slug,
        "title": match.title,
        "region": match.region,
        "bo_x": match.bo_x,
        "team_a": match.team_a_t1_name,
        "team_b": match.team_b_t1_name,
        "team_a_raw": match.team_a,
        "team_b_raw": match.team_b,
        "game_start_time": match.game_start_time,
    }


def backfill(
    regions: tuple[str, ...] = ("americas", "emea", "pacific", "china", "international"),
    include_liquidity: bool = True,
    verbose: bool = True,
) -> dict:
    """One-shot historical mine: every T1 match Polymarket has ever listed,
    snapshotted at T-24h/T-12h/pre-match (and the decider map's price, when
    the series went the distance), using price-history -- no live polling
    needed since these matches already happened."""
    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = {"events_scanned": 0, "t1_matches_found": 0, "matches_with_snapshots": 0, "rows_written": 0}

    with PolymarketFetcher() as fetcher:
        matches = discover_t1_matches(fetcher, verbose=verbose)
        summary["t1_matches_found"] = len(matches)
        matches = [m for m in matches if m.region in regions]

        with JSONL_PATH.open("w", encoding="utf-8") as f:
            for i, match in enumerate(matches, 1):
                rows = _market_rows(fetcher, match, "match_winner", match.match_winner)
                for mm in match.map_markets:
                    rows.extend(_market_rows(fetcher, match, f"map{mm.map_num}", mm))

                if include_liquidity and rows:
                    trades = fetch_trades(fetcher, match.match_winner.condition_id)
                    for row in rows:
                        row.update(trades_in_window(trades, row["anchor_target_ts"]))

                for row in rows:
                    f.write(json.dumps(row) + "\n")
                summary["rows_written"] += len(rows)
                if rows:
                    summary["matches_with_snapshots"] += 1
                if verbose and i % 10 == 0:
                    print(f"  [{i}/{len(matches)}] matches processed, {summary['rows_written']} rows so far", file=sys.stderr)

    return summary


def export_excel(jsonl_path: Path = JSONL_PATH, xlsx_path: Path = XLSX_PATH) -> int:
    import pandas as pd

    if not jsonl_path.exists():
        raise FileNotFoundError(f"{jsonl_path} does not exist yet -- run `polymarket backfill` first")

    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{jsonl_path} has no rows yet")

    long_df = pd.DataFrame(rows)
    long_df = long_df.sort_values(["event_id", "market", "anchor"])

    match_cols = ["event_id", "slug", "title", "region", "bo_x", "team_a", "team_b", "game_start_time"]
    wide = long_df.drop_duplicates("event_id")[match_cols].set_index("event_id")

    metric_cols = [
        "price_team_a", "price_team_b", "vig", "fair_prob_team_a", "fair_prob_team_b",
        "gap_seconds_a", "gap_seconds_b", "trades_n", "trades_size", "trades_notional",
    ]
    combos = long_df[["market", "anchor"]].drop_duplicates().itertuples(index=False)
    for market, anchor in combos:
        subset = long_df[(long_df["market"] == market) & (long_df["anchor"] == anchor)].set_index("event_id")
        prefix = f"{market}_{anchor}"
        for col in metric_cols:
            if col in subset.columns:
                wide[f"{prefix}_{col}"] = subset[col]

    vig_cols = [c for c in wide.columns if c.endswith("_vig")]
    wide["avg_vig"] = wide[vig_cols].mean(axis=1, skipna=True) if vig_cols else None

    wide = wide.reset_index()

    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        wide.to_excel(writer, sheet_name="Polymarket Odds (wide)", index=False)
        long_df.to_excel(writer, sheet_name="Polymarket Odds (long)", index=False)

    return len(wide)


# ---------------------------------------------------------------------------
# Forward-looking live poller -- captures genuine order-book depth, which
# (unlike price) can't be reconstructed after the fact. Run this on a
# schedule (see scripts/run_polymarket_poll.sh); the historical backfill
# above already covers everything that's already happened.
# ---------------------------------------------------------------------------

POLL_LOOKAHEAD_HOURS = 30  # start tracking a match this far before it's scheduled to start
POLL_BOOK_BAND_PCT = 0.05  # "near-mid" depth = size resting within +/-5% of the midpoint


def _state_path(event_id: str) -> Path:
    return STATE_DIR / f"{event_id}.json"


def _load_state(event_id: str) -> dict | None:
    path = _state_path(event_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(event_id: str, state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _state_path(event_id).write_text(json.dumps(state, indent=2), encoding="utf-8")


def _load_finalized_ids() -> set[str]:
    if not FINALIZED_IDS_PATH.exists():
        return set()
    return set(json.loads(FINALIZED_IDS_PATH.read_text(encoding="utf-8")))


def _save_finalized_ids(ids: set[str]) -> None:
    FINALIZED_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINALIZED_IDS_PATH.write_text(json.dumps(sorted(ids)), encoding="utf-8")


def fetch_book(fetcher: PolymarketFetcher, token_id: str) -> dict | None:
    return fetcher.get_clob("/book", params={"token_id": token_id})


def book_depth_near_mid(book: dict | None, band_pct: float = POLL_BOOK_BAND_PCT) -> dict:
    """Total resting size within `band_pct` of the book's own midpoint, plus
    top-of-book bid/ask/spread -- the genuine real-time liquidity signal that
    has no historical equivalent."""
    if not book or not book.get("bids") or not book.get("asks"):
        return {"best_bid": None, "best_ask": None, "spread": None, "depth_near_mid": None}
    bids = sorted(((float(b["price"]), float(b["size"])) for b in book["bids"]), key=lambda x: -x[0])
    asks = sorted(((float(a["price"]), float(a["size"])) for a in book["asks"]), key=lambda x: x[0])
    best_bid, best_ask = bids[0][0], asks[0][0]
    mid = (best_bid + best_ask) / 2
    lo, hi = mid * (1 - band_pct), mid * (1 + band_pct)
    depth = sum(sz for px, sz in bids if px >= lo) + sum(sz for px, sz in asks if px <= hi)
    return {"best_bid": best_bid, "best_ask": best_ask, "spread": best_ask - best_bid, "depth_near_mid": depth}


def _snapshot_market_live(fetcher: PolymarketFetcher, market: PMMarket) -> dict:
    book_a = fetch_book(fetcher, market.clob_token_ids[0])
    book_b = fetch_book(fetcher, market.clob_token_ids[1])
    depth_a, depth_b = book_depth_near_mid(book_a), book_depth_near_mid(book_b)
    pa, pb = depth_a["best_ask"], depth_b["best_ask"]  # cost to actually buy each side now
    vig = (pa + pb - 1) if (pa is not None and pb is not None) else None
    return {
        "captured_at": _now_iso(),
        "price_team_a": pa,
        "price_team_b": pb,
        "vig": vig,
        "book_team_a": depth_a,
        "book_team_b": depth_b,
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def poll_once(lookahead_hours: int = POLL_LOOKAHEAD_HOURS, verbose: bool = True) -> dict:
    """One poll cycle: snapshot live price+order-book depth for every T1 match
    within `lookahead_hours` of its scheduled start (or already live), then
    finalize any tracked match whose event has since closed -- picking the
    stored snapshot nearest each anchor (T-24h/T-12h/pre-match/decider) and
    appending it to the same JSONL the backfill writes to."""
    summary = {"snapshotted": 0, "finalized": 0}
    finalized_ids = _load_finalized_ids()
    now_ts = int(datetime.now(timezone.utc).timestamp())

    with PolymarketFetcher() as fetcher:
        roster = load_t1_roster()
        open_events = discover_events_open_only(fetcher, verbose=verbose)
        open_matches = [m for m in (parse_pm_match(e, roster) for e in open_events) if m]

        for match in open_matches:
            start_ts = _iso_to_ts(match.game_start_time)
            if start_ts is None or now_ts < start_ts - lookahead_hours * 3600:
                continue
            state = _load_state(match.event_id) or {"event_id": match.event_id, "match_meta": _base_row(match), "snapshots": []}
            snap = {"market_snapshots": {}}
            snap["market_snapshots"]["match_winner"] = _snapshot_market_live(fetcher, match.match_winner)
            for mm in match.map_markets:
                snap["market_snapshots"][f"map{mm.map_num}"] = _snapshot_market_live(fetcher, mm)
            snap["captured_at"] = _now_iso()
            state["snapshots"].append(snap)
            _save_state(match.event_id, state)
            summary["snapshotted"] += 1
            if verbose:
                print(f"  snapshot {match.title} ({len(state['snapshots'])} points so far)", file=sys.stderr)

        # Finalize: any state file whose event has since closed.
        JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with JSONL_PATH.open("a", encoding="utf-8") as jsonl_f:
            for state_file in STATE_DIR.glob("*.json"):
                event_id = state_file.stem
                if event_id in finalized_ids:
                    continue
                event = fetcher.get_gamma(f"/events/{event_id}")
                if not event or not event.get("closed"):
                    continue
                match = parse_pm_match(event, roster)
                state = _load_state(event_id)
                if match is None or state is None or not state["snapshots"]:
                    finalized_ids.add(event_id)
                    state_file.unlink(missing_ok=True)
                    continue

                start_ts = _iso_to_ts(match.game_start_time)
                decider = decider_target(match)
                targets = {label: start_ts - int(off.total_seconds()) for label, off in ANCHOR_OFFSETS.items()} if start_ts else {}
                if decider is not None:
                    targets["decider"] = decider[0]

                for market_label in ["match_winner"] + [f"map{mm.map_num}" for mm in match.map_markets]:
                    points = [
                        (int(datetime.fromisoformat(s["captured_at"]).timestamp()), s["market_snapshots"].get(market_label))
                        for s in state["snapshots"]
                        if s["market_snapshots"].get(market_label)
                    ]
                    if not points:
                        continue
                    for anchor_label, target_ts in targets.items():
                        if anchor_label == "decider" and market_label != "match_winner":
                            continue
                        ts, live_snap = min(points, key=lambda p: abs(p[0] - target_ts))
                        row = {
                            **_base_row(match),
                            "market": market_label if anchor_label != "decider" else f"map{decider[1]}_decider",
                            "anchor": anchor_label,
                            "anchor_target_ts": target_ts,
                            "actual_ts_a": ts,
                            "actual_ts_b": ts,
                            "gap_seconds_a": ts - target_ts,
                            "gap_seconds_b": ts - target_ts,
                            "price_team_a": live_snap["price_team_a"],
                            "price_team_b": live_snap["price_team_b"],
                            "vig": live_snap["vig"],
                            "source": "live_poll",
                            "book_depth_near_mid_a": live_snap["book_team_a"]["depth_near_mid"],
                            "book_depth_near_mid_b": live_snap["book_team_b"]["depth_near_mid"],
                            "book_spread_a": live_snap["book_team_a"]["spread"],
                            "book_spread_b": live_snap["book_team_b"]["spread"],
                        }
                        jsonl_f.write(json.dumps(row) + "\n")

                finalized_ids.add(event_id)
                state_file.unlink(missing_ok=True)
                summary["finalized"] += 1

    _save_finalized_ids(finalized_ids)
    return summary


def discover_events_open_only(fetcher: PolymarketFetcher, verbose: bool = True) -> list[dict]:
    events: list[dict] = []
    offset = 0
    while True:
        page = fetcher.get_gamma("/events", params={"tag_slug": "valorant", "closed": "false", "limit": 100, "offset": offset})
        if not page:
            break
        events.extend(page)
        if len(page) < 100:
            break
        offset += 100
    if verbose:
        print(f"  discover (open only): {len(events)} events", file=sys.stderr)
    return events
