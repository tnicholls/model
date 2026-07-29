"""Data schemas for scraped vlr.gg entities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class AgentUsage:
    agent: str
    pct: float | None


@dataclass
class PlayerStat:
    """One row from a vlr.gg stats table (event stats page or global /stats page)."""

    player_id: int
    player_alias: str
    team_or_country: str | None
    flag: str | None
    agents: list[AgentUsage]
    maps: int | None
    rounds: int | None
    rating: float | None
    acs: float | None
    kd: float | None
    kast_pct: float | None
    adr: float | None
    kpr: float | None
    apr: float | None
    fkfd: float | None
    fkpr: float | None
    fdpr: float | None
    hs_pct: float | None
    clutch_pct: float | None
    clutches_won: int | None
    clutches_played: int | None
    kmax: int | None
    kmax_match_url: str | None
    kills: int | None
    deaths: int | None
    assists: int | None
    first_kills: int | None
    first_deaths: int | None
    source: str = ""  # e.g. "event:2283" or "global"
    filters: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StandingEntry:
    event_id: int
    place_display: str
    place_rank: int | None
    prize: str | None
    team_id: int | None
    team_name: str | None
    team_country: str | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RosterPlayer:
    player_id: int
    alias: str
    real_name: str | None
    country: str | None
    role: str | None  # None for players, e.g. "manager"/"coach" for staff
    is_captain: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MatchSummary:
    """One row from a vlr.gg match list (global schedule/results, or an event's match tab)."""

    match_id: int
    match_url: str
    date_display: str | None
    time_display: str | None
    team1: str | None
    team1_flag: str | None
    team1_score: int | None
    team1_won: bool
    team2: str | None
    team2_flag: str | None
    team2_score: int | None
    team2_won: bool
    status: str | None  # "Completed", "LIVE", "Upcoming"
    eta: str | None  # e.g. "5h 20m", "1h 10m", "10mo"
    event_name: str | None
    series: str | None  # round/stage label, e.g. "Group Stage-Week 2"
    source: str = ""  # e.g. "event:2283" or "global:results"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BetLine:
    """One bookmaker's odds for one team in one match, from a match page's Betting widget.

    vlr.gg shows this widget in two shapes depending on match state:
      * pre-match / live: both teams' decimal odds side by side -> one BetLine per team,
        with `implied_prob_fair` computed by de-vigging the pair.
      * post-match (mod-post-odds): only the *winning* team's odds survive, shown as a
        "$100 -> $payout" line -> a single BetLine with `stake`/`payout` set and
        `implied_prob_fair` left None (the losing side's odds are no longer on the page).
    """

    match_id: int
    bookmaker: str | None
    bet_url: str | None
    stage: str | None  # "pre-match", "live", or "post-match"
    team: str | None
    team_tag: str | None
    decimal_odds: float | None
    stake: float | None  # post-match only, e.g. 100.0
    payout: float | None  # post-match only, e.g. 178.0
    implied_prob_raw: float | None  # 1 / decimal_odds
    implied_prob_fair: float | None  # de-vigged, only when both sides of the pair are known
    source: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentMapPickRate:
    """One (map, agent) pick-rate row from an event's `/event/agents/{id}/{slug}`
    page. `map_name` is "All" for the table's event-wide aggregate row."""

    event_id: int
    map_name: str
    games: int | None
    atk_win_pct: float | None
    def_win_pct: float | None
    agent: str
    pick_rate_pct: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlayerMapStat:
    """One (player, map) row from a match page's per-map overview table
    (`div.ovw-table` inside each `div.vm-stats-game`). Used to build the
    agent-proficiency map-strength features -- see agent_proficiency.py.

    `team_tag` is the raw short tag as rendered on the page (e.g. "SEN");
    resolving it to a team_id is left to the caller (needs TEAM_TAGS_PATH,
    which parse.py deliberately doesn't touch -- see veto._resolve_veto_team
    for why fuzzy name matching isn't safe here).
    """

    match_id: int
    map_id: int
    map_name: str
    match_date_utc: str | None  # "YYYY-MM-DD HH:MM:SS", UTC
    patch: str | None  # e.g. "12.08"
    team1_id: int | None
    team2_id: int | None
    rounds_played: int | None  # total rounds on this map (both teams' scores summed)
    player_id: int
    player_alias: str
    team_tag: str | None
    agent: str | None
    rating: float | None  # Rating 2.0, "both sides" value

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TeamInfo:
    team_id: int
    name: str
    tag: str | None
    country: str | None
    links: list[str]
    roster: list[RosterPlayer]
    staff: list[RosterPlayer]

    def to_dict(self) -> dict:
        return {
            **{k: v for k, v in asdict(self).items() if k not in ("roster", "staff")},
            "roster": [p.to_dict() for p in self.roster],
            "staff": [p.to_dict() for p in self.staff],
        }
