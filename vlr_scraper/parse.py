"""HTML -> structured data for vlr.gg pages.

Covers three page families:
  * stats tables (`table#st-table`), shared verbatim between the global
    `/stats` leaderboard and an event's `/event/stats/{id}/{slug}` page
  * an event's standings table (`div.wf-ptable--standings`) on
    `/event/{id}/{slug}`
  * a team's roster/info on `/team/{id}/{slug}`
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag

from .models import AgentMapPickRate, AgentUsage, BetLine, MatchSummary, PlayerStat, RosterPlayer, StandingEntry, TeamInfo

_PLAYER_HREF_RE = re.compile(r"/player/(\d+)/")
_TEAM_HREF_RE = re.compile(r"/team/(\d+)/")
_PLACE_RANK_RE = re.compile(r"(\d+)")


def _clean(text: str | None) -> str | None:
    if text is None:
        return None
    text = text.strip()
    return text or None


def _num(text: str | None, cast=float):
    text = _clean(text)
    if text is None or text == "-":
        return None
    text = text.rstrip("%").replace(",", "").lstrip("$")
    try:
        return cast(text)
    except ValueError:
        return None


def _agent_name(img_src: str) -> str:
    # "/img/vlr/game/agents/neon.png" -> "neon"
    return img_src.rsplit("/", 1)[-1].removesuffix(".png")


def _flag_code(container: Tag | None) -> str | None:
    """Country code from a `<i/span class="flag mod-xx">` descendant, e.g. "br"."""
    if container is None:
        return None
    flag_tag = container.find(class_="flag")
    if flag_tag is None:
        return None
    flag_classes = [c for c in flag_tag.get("class", []) if c.startswith("mod-")]
    return flag_classes[0].removeprefix("mod-") if flag_classes else None


def _direct_text(tag: Tag | None) -> str | None:
    """First non-blank text found directly under `tag` (not inside a nested tag).

    Several vlr.gg cells mix a label's own text with a nested sub-label div,
    e.g. `<div class="text-of">NRG<div class="ge-text-light">USA</div></div>`
    or, with the text *after* the nested tag instead of before it,
    `<div class="match-item-event text-of"><div class="...-series">A</div>Group Stage</div>`.
    Scanning direct children in order (skipping nested Tags and blank strings)
    handles both orderings.
    """
    if tag is None:
        return None
    for child in tag.contents:
        if isinstance(child, str):
            text = child.strip()
            if text:
                return text
    return None


def parse_stats_table(html: str, source: str = "", filters: dict | None = None) -> list[PlayerStat]:
    """Parse a vlr.gg player-stats table.

    `source` is a free-form label describing where this came from, e.g.
    "event:2283" or "global" -- stored on each row for provenance.
    `filters` records the query params used to fetch the page (side, role,
    agent, map_id, min_rounds, etc.) since the table's contents depend on them.
    """
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id="st-table")
    if table is None:
        return []

    rows: list[PlayerStat] = []
    for tr in table.find("tbody").find_all("tr", recursive=False):
        cells = {td.get("data-col"): td for td in tr.find_all("td", attrs={"data-col": True})}

        player_td = tr.find("td", class_="mod-player")
        link = player_td.find("a") if player_td else None
        if link is None:
            continue
        player_match = _PLAYER_HREF_RE.search(link.get("href", ""))
        if not player_match:
            continue
        player_id = int(player_match.group(1))
        alias = _clean(link.find(class_="st-pl-name").get_text()) if link.find(class_="st-pl-name") else None
        team_or_country = None
        country_div = link.find(class_="st-pl-country")
        if country_div:
            team_or_country = _clean(country_div.get_text())
        flag = _flag_code(link)

        agents: list[AgentUsage] = []
        agents_td = cells.get("agents")
        if agents_td:
            for span in agents_td.find_all("span", class_="st-agent"):
                img = span.find("img")
                pct_span = span.find(class_="st-agent-n")
                if img is None:
                    continue
                agents.append(AgentUsage(agent=_agent_name(img.get("src", "")), pct=_num(pct_span.get_text() if pct_span else None)))

        def cell_num(col: str, cast=float):
            td = cells.get(col)
            return _num(td.get_text() if td else None, cast=cast)

        clutches_won = clutches_played = None
        cl_td = cells.get("cl")
        if cl_td:
            x = cl_td.find(class_="st-cl-x")
            y = cl_td.find(class_="st-cl-y")
            clutches_won = _num(x.get_text() if x else None, cast=int)
            clutches_played = _num(y.get_text() if y else None, cast=int)

        kmax = kmax_url = None
        kmax_td = cells.get("kmax")
        if kmax_td:
            kmax_link = kmax_td.find("a")
            if kmax_link:
                kmax = _num(kmax_link.get_text(), cast=int)
                kmax_url = kmax_link.get("href")
            else:
                kmax = cell_num("kmax", int)

        rows.append(
            PlayerStat(
                player_id=player_id,
                player_alias=alias or "",
                team_or_country=team_or_country,
                flag=flag,
                agents=agents,
                maps=cell_num("maps", int),
                rounds=cell_num("rnd", int),
                rating=cell_num("rating2"),
                acs=cell_num("acs"),
                kd=cell_num("kd"),
                kast_pct=cell_num("kast"),
                adr=cell_num("adr"),
                kpr=cell_num("kpr"),
                apr=cell_num("apr"),
                fkfd=cell_num("fkfd"),
                fkpr=cell_num("fbpr"),
                fdpr=cell_num("fdpr"),
                hs_pct=cell_num("hsp"),
                clutch_pct=cell_num("clp"),
                clutches_won=clutches_won,
                clutches_played=clutches_played,
                kmax=kmax,
                kmax_match_url=kmax_url,
                kills=cell_num("k", int),
                deaths=cell_num("d", int),
                assists=cell_num("a", int),
                first_kills=cell_num("fk", int),
                first_deaths=cell_num("fd", int),
                source=source,
                filters=filters or {},
            )
        )
    return rows


def parse_event_standings(html: str, event_id: int) -> list[StandingEntry]:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("div", class_="wf-ptable--standings")
    if table is None:
        return []

    entries: list[StandingEntry] = []
    data_rows = table.find_all("div", class_="row", recursive=False)[1:]  # skip header row
    for row in data_rows:
        cells = row.find_all("div", class_="cell", recursive=False)
        if len(cells) < 3:
            continue
        place_display = _clean(cells[0].get_text(" ")) or ""
        place_display = re.sub(r"\s+", "", place_display)
        rank_match = _PLACE_RANK_RE.search(place_display)
        place_rank = int(rank_match.group(1)) if rank_match else None

        prize = _clean(cells[1].get_text())

        team_cell = cells[2]
        link = team_cell.find("a")
        team_id = team_name = team_country = None
        if link:
            team_match = _TEAM_HREF_RE.search(link.get("href", ""))
            team_id = int(team_match.group(1)) if team_match else None
            name_div = link.find(class_="text-of")
            if name_div:
                country_div = name_div.find(class_="ge-text-light")
                team_country = _clean(country_div.get_text()) if country_div else None
                team_name = _direct_text(name_div)

        entries.append(
            StandingEntry(
                event_id=event_id,
                place_display=place_display,
                place_rank=place_rank,
                prize=prize,
                team_id=team_id,
                team_name=team_name,
                team_country=team_country,
            )
        )
    return entries


def parse_team(html: str, team_id: int) -> TeamInfo:
    soup = BeautifulSoup(html, "lxml")
    header = soup.find("div", class_="team-header")

    name = tag = country = None
    links: list[str] = []
    if header:
        name_tag = header.find("h1", class_="wf-title")
        name = _clean(name_tag.get_text()) if name_tag else None
        tag_tag = header.find("h2", class_="team-header-tag")
        tag = _clean(tag_tag.get_text()) if tag_tag else None
        country_div = header.find(class_="team-header-country")
        if country_div:
            country = _clean(country_div.get_text())
        links_div = header.find(class_="team-header-links")
        if links_div:
            links = [a.get("href") for a in links_div.find_all("a") if a.get("href")]

    roster: list[RosterPlayer] = []
    staff: list[RosterPlayer] = []
    for item in soup.find_all("div", class_="team-roster-item"):
        link = item.find("a")
        if link is None:
            continue
        player_match = _PLAYER_HREF_RE.search(link.get("href", ""))
        if not player_match:
            continue
        player_id = int(player_match.group(1))

        alias_div = item.find(class_="team-roster-item-name-alias")
        alias = _clean(alias_div.get_text()) if alias_div else ""
        is_captain = bool(alias_div and alias_div.find("i", class_="fa-star"))

        real_name_div = item.find(class_="team-roster-item-name-real")
        real_name = _clean(real_name_div.get_text()) if real_name_div else None

        player_country = _flag_code(item)

        role_div = item.find(class_="team-roster-item-name-role")
        role = _clean(role_div.get_text()) if role_div else None

        player = RosterPlayer(
            player_id=player_id,
            alias=alias or "",
            real_name=real_name,
            country=player_country,
            role=role,
            is_captain=is_captain,
        )
        (staff if role else roster).append(player)

    return TeamInfo(
        team_id=team_id,
        name=name or "",
        tag=tag,
        country=country,
        links=links,
        roster=roster,
        staff=staff,
    )


def parse_agent_pick_rates(html: str, event_id: int) -> list[AgentMapPickRate]:
    """Parse an event's `/event/agents/{id}/{slug}` "Agent Pick Rates and
    Compositions" page: one row per map (plus an event-wide "All" row) x one
    column per agent, giving that agent's pick rate on that map.

    NOT covered here or anywhere else in this codebase: rating (R2.0) broken
    down by agent AND map together (e.g. "Sova's average rating on Haven").
    The global /stats page (see cmd_stats in cli.py) has `agent` and `map_id`
    query params that look like they'd give this, but they're a dead end for
    plain HTTP requests -- confirmed by testing both directly: requesting
    /stats with agent=sova or map_id=haven returns byte-identical results to
    the unfiltered page. Only the coarse filters (tier, region, span) are
    server-rendered from the query string; the granular ones (agent, map_id,
    side, role) only take effect through the page's own "Apply Filter"
    button/JS, which a bare `requests.get()` doesn't trigger -- likely
    session- or Referer-gated, not just a param-naming issue (both `agent`
    and `map_id` were tested and both no-op).

    To actually get this stat, one of:
      (a) proxy it: for a given map, take players whose dominant pick on
          that map is agent X (>~50% from parse_stats_table's `agents` list)
          and use their overall rating as an estimate -- fast, reuses
          existing scraping, but conflates player skill with agent effect;
      (b) do it for real via browser automation driving the actual
          Apply-Filter click per (map, agent) combo -- exact, but one real
          page load per combo (7 maps x 29 agents = ~200 loads minimum).
    Neither is implemented. (a) was proposed and explicitly declined in
    favor of shipping pick-rate-by-player/team/map first; (b) wasn't
    attempted at all.
    """
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="mod-pr-global")
    if table is None:
        return []

    rows = table.find_all("tr")
    if len(rows) < 2:
        return []

    agent_names: list[str | None] = []
    for th in rows[0].find_all("th")[4:]:
        img = th.find("img")
        agent_names.append(_agent_name(img.get("src", "")) if img else None)

    entries: list[AgentMapPickRate] = []
    for tr in rows[1:]:
        cells = tr.find_all("td")
        if len(cells) < 4:
            continue
        map_name = _direct_text(cells[0]) or "All"
        games = _num(cells[1].get_text(), cast=int)
        atk_win_pct = _num(cells[2].get_text())
        def_win_pct = _num(cells[3].get_text())
        for agent, td in zip(agent_names, cells[4:]):
            if agent is None:
                continue
            entries.append(
                AgentMapPickRate(
                    event_id=event_id,
                    map_name=map_name,
                    games=games,
                    atk_win_pct=atk_win_pct,
                    def_win_pct=def_win_pct,
                    agent=agent,
                    pick_rate_pct=_num(td.get_text()),
                )
            )
    return entries


_MATCH_HREF_RE = re.compile(r"^/(\d+)/")


def _parse_match_item(a: Tag, date_display: str | None, source: str) -> MatchSummary | None:
    href = a.get("href", "")
    match = _MATCH_HREF_RE.match(href)
    if not match:
        return None
    match_id = int(match.group(1))

    time_div = a.find(class_="match-item-time")
    time_display = _clean(time_div.get_text()) if time_div else None

    team_divs = a.find_all(class_="match-item-vs-team")
    teams: list[dict] = []
    for team_div in team_divs:
        name_div = team_div.find(class_="match-item-vs-team-name")
        text_of = name_div.find(class_="text-of") if name_div else None
        score_div = team_div.find(class_="match-item-vs-team-score")
        teams.append(
            {
                "name": _clean(text_of.get_text()) if text_of else None,
                "flag": _flag_code(text_of),
                "score": _num(score_div.get_text() if score_div else None, cast=int),
                "won": "mod-winner" in (team_div.get("class") or []),
            }
        )
    while len(teams) < 2:
        teams.append({"name": None, "flag": None, "score": None, "won": False})

    eta_container = a.find(class_="match-item-eta")
    status = eta = None
    if eta_container:
        status_div = eta_container.find(class_="ml-status")
        status = _clean(status_div.get_text()) if status_div else None
        eta_div = eta_container.find(class_="ml-eta")
        eta = _clean(eta_div.get_text()) if eta_div else None

    event_div = a.find(class_="match-item-event")
    event_name = series = None
    if event_div:
        series_div = event_div.find(class_="match-item-event-series")
        series = _clean(series_div.get_text()) if series_div else None
        event_name = _direct_text(event_div)

    return MatchSummary(
        match_id=match_id,
        match_url=href,
        date_display=date_display,
        time_display=time_display,
        team1=teams[0]["name"],
        team1_flag=teams[0]["flag"],
        team1_score=teams[0]["score"],
        team1_won=teams[0]["won"],
        team2=teams[1]["name"],
        team2_flag=teams[1]["flag"],
        team2_score=teams[1]["score"],
        team2_won=teams[1]["won"],
        status=status,
        eta=eta,
        event_name=event_name,
        series=series,
        source=source,
    )


def parse_matches(html: str, source: str = "") -> list[MatchSummary]:
    """Parse a vlr.gg match list: global `/matches` (schedule), `/matches/results`,
    or an event's `/event/matches/{id}/{slug}` tab -- all three share the same
    `wf-label.mod-large` (date header) + `a.match-item` (one match) markup.
    """
    soup = BeautifulSoup(html, "lxml")
    matches: list[MatchSummary] = []
    current_date: str | None = None
    for el in soup.find_all(["div", "a"]):
        classes = el.get("class") or []
        if el.name == "div" and "wf-label" in classes and "mod-large" in classes:
            current_date = _direct_text(el)
        elif el.name == "a" and "match-item" in classes:
            parsed = _parse_match_item(el, current_date, source)
            if parsed:
                matches.append(parsed)
    return matches


def _bookmaker_name(card: Tag) -> str | None:
    img = card.find("img")
    if img is None:
        return None
    logo_classes = [c for c in img.get("class", []) if c.startswith("mod-")]
    return logo_classes[0].removeprefix("mod-") if logo_classes else None


def _parse_pair_half(half: Tag) -> tuple[str | None, str | None, float | None]:
    name_span = half.find(class_="match-bet-item-team-name")
    tag_span = half.find(class_="match-bet-item-team-tag")
    odds_span = half.find(class_="match-bet-item-odds")
    name = _clean(name_span.get_text()) if name_span else None
    tag = _clean(tag_span.get_text()) if tag_span else None
    odds = _num(odds_span.get_text() if odds_span else None)
    return name, tag, odds


def parse_match_odds(html: str, match_id: int, source: str = "") -> list[BetLine]:
    """Parse a match page's "Betting" widget into one row per (bookmaker, team).

    Never fetch the `/rr/bet/...` affiliate links found here -- robots.txt disallows
    `/rr/`, and the odds themselves are already rendered on this page, so `bet_url` is
    recorded purely for reference.
    """
    soup = BeautifulSoup(html, "lxml")
    lines: list[BetLine] = []

    for card in soup.find_all("a", class_="match-bet-item"):
        classes = card.get("class") or []
        if "mod-noodds" in classes:
            continue  # generic sponsor promo card, carries no actual odds

        bookmaker = _bookmaker_name(card)
        bet_url = card.get("href")

        if "mod-post-odds" in classes:
            msg_div = card.find(class_="match-bet-item-return-msg")
            short_div = card.find(class_="match-bet-item-return-short")
            if msg_div is None or short_div is None:
                continue
            odds_amounts = msg_div.find_all(class_="match-bet-item-odds")
            stake = _num(odds_amounts[0].get_text()) if len(odds_amounts) > 0 else None
            payout = _num(odds_amounts[1].get_text()) if len(odds_amounts) > 1 else None
            team_span = short_div.find(class_="match-bet-item-teamzzz")
            team = _clean(team_span.get_text()) if team_span else None
            decimal_span = short_div.find(class_="match-bet-item-odds")
            decimal_odds = _num(decimal_span.get_text()) if decimal_span else None

            lines.append(
                BetLine(
                    match_id=match_id,
                    bookmaker=bookmaker,
                    bet_url=bet_url,
                    stage="post-match",
                    team=team,
                    team_tag=None,
                    decimal_odds=decimal_odds,
                    stake=stake,
                    payout=payout,
                    implied_prob_raw=(1 / decimal_odds) if decimal_odds else None,
                    implied_prob_fair=None,  # losing side's odds aren't on the page anymore
                    source=source,
                )
            )
            continue

        halves = card.find_all("div", class_="match-bet-item-half")
        half1 = next((h for h in halves if "mod-1" in (h.get("class") or [])), None)
        half2 = next((h for h in halves if "mod-2" in (h.get("class") or [])), None)
        if half1 is None or half2 is None:
            continue

        team1, tag1, odds1 = _parse_pair_half(half1)
        team2, tag2, odds2 = _parse_pair_half(half2)

        note_div = card.find(class_="match-bet-item-note")
        note_text = _clean(note_div.get_text()) if note_div else None
        stage = note_text.lower() if note_text else "pre-match"

        fair1 = fair2 = None
        if odds1 and odds2:
            inv1, inv2 = 1 / odds1, 1 / odds2
            total = inv1 + inv2
            fair1, fair2 = inv1 / total, inv2 / total

        lines.append(
            BetLine(
                match_id=match_id,
                bookmaker=bookmaker,
                bet_url=bet_url,
                stage=stage,
                team=team1,
                team_tag=tag1,
                decimal_odds=odds1,
                stake=None,
                payout=None,
                implied_prob_raw=(1 / odds1) if odds1 else None,
                implied_prob_fair=fair1,
                source=source,
            )
        )
        lines.append(
            BetLine(
                match_id=match_id,
                bookmaker=bookmaker,
                bet_url=bet_url,
                stage=stage,
                team=team2,
                team_tag=tag2,
                decimal_odds=odds2,
                stake=None,
                payout=None,
                implied_prob_raw=(1 / odds2) if odds2 else None,
                implied_prob_fair=fair2,
                source=source,
            )
        )
    return lines


_EVENT_HREF_RE = re.compile(r"/event/(\d+)/")


def parse_events_list(html: str) -> list[dict]:
    """Parse a `/events` listing page into [{event_id, name, status}, ...]."""
    soup = BeautifulSoup(html, "lxml")
    out: list[dict] = []
    for a in soup.find_all("a", class_="event-item"):
        m = _EVENT_HREF_RE.search(a.get("href", ""))
        if not m:
            continue
        title_div = a.find(class_="event-item-title")
        status_div = a.find(class_="event-item-desc-item-status")
        out.append(
            {
                "event_id": int(m.group(1)),
                "name": _clean(title_div.get_text()) if title_div else None,
                "status": _clean(status_div.get_text()) if status_div else None,
            }
        )
    return out
