#!/usr/bin/env python3
"""CLI wrapper around vlr_scraper.

Examples:
    python cli.py stats --event-id 2283 --out data/raw/event_2283_stats.json
    python cli.py stats --global --region na --timespan 60d --min-rounds 200 \
        --out data/raw/global_stats.json
    python cli.py standings --event-id 2283 --out data/raw/event_2283_standings.json
    python cli.py team --team-id 1034 --out data/raw/team_1034.json
    python cli.py matches --event-id 2283 --out data/raw/event_2283_matches.json
    python cli.py matches --status results --pages 3 --out data/raw/global_results.json
    python cli.py matches --status upcoming --out data/raw/global_upcoming.json

    python cli.py odds --match-id 701063 --out data/raw/match_701063_odds.json
    python cli.py odds --from-matches data/raw/event_2283_matches.json \
        --out data/raw/event_2283_odds.json

    python cli.py flatten stats --input data/raw/event_2283_stats.json \
        --out data/processed/event_2283_stats.csv
    python cli.py flatten standings --input data/raw/event_2283_standings.json \
        --out data/processed/event_2283_standings.csv
    python cli.py flatten matches --input data/raw/event_2283_matches.json \
        --out data/processed/event_2283_matches.csv
    python cli.py flatten odds --input data/raw/event_2283_odds.json \
        --out data/processed/event_2283_odds.csv
    python cli.py flatten team --input data/raw/team_1034.json \
        --out data/processed/team_1034_roster.csv

    python cli.py closing poll     # snapshot upcoming odds + finalize newly-completed matches
    python cli.py closing export   # rebuild data/closing_lines/valorant_closing_lines.xlsx
    python cli.py closing run      # both, in one call (what the scheduled job runs)
"""

from __future__ import annotations

import argparse
import sys

from vlr_scraper import closing_lines
from vlr_scraper.fetch import Fetcher
from vlr_scraper.models import BetLine, MatchSummary, PlayerStat, RosterPlayer, StandingEntry, TeamInfo
from vlr_scraper.parse import parse_event_standings, parse_match_odds, parse_matches, parse_stats_table, parse_team
from vlr_scraper.storage import (
    load_json,
    matches_to_rows,
    odds_to_rows,
    player_stats_to_rows,
    roster_to_rows,
    save_json,
    standings_to_rows,
    write_csv,
)


def cmd_stats(args: argparse.Namespace) -> None:
    params = {
        "side": args.side,
        "role": args.role,
        "agent": args.agent,
        "map_id": args.map_id,
        "min_rounds": args.min_rounds,
    }
    if args.sort:
        params["sort"] = args.sort
        params["dir"] = args.dir

    if args.event_id is not None:
        path = f"/event/stats/{args.event_id}"
        source = f"event:{args.event_id}"
    else:
        path = "/stats"
        params.update(
            {
                "event_group_id": "all",
                "event_id": "all",
                "region": args.region,
                "min_rating": args.min_rating,
                "timespan": args.timespan,
            }
        )
        source = "global"

    with Fetcher() as fetcher:
        html = fetcher.get_html(path, params=params)

    rows = parse_stats_table(html, source=source, filters=params)
    print(f"Parsed {len(rows)} player rows from {source}", file=sys.stderr)
    save_json([r.to_dict() for r in rows], args.out)
    print(f"Wrote {args.out}", file=sys.stderr)


def cmd_matches(args: argparse.Namespace) -> None:
    all_matches = []
    with Fetcher() as fetcher:
        if args.event_id is not None:
            source = f"event:{args.event_id}"
            html = fetcher.get_html(f"/event/matches/{args.event_id}", params={"series_id": "all"})
            all_matches.extend(parse_matches(html, source=source))
        else:
            source = f"global:{args.status}"
            path = "/matches" if args.status == "upcoming" else "/matches/results"
            for page in range(args.page, args.page + args.pages):
                html = fetcher.get_html(path, params={"page": page})
                all_matches.extend(parse_matches(html, source=source))

    print(f"Parsed {len(all_matches)} matches from {source}", file=sys.stderr)
    save_json([m.to_dict() for m in all_matches], args.out)
    print(f"Wrote {args.out}", file=sys.stderr)


def cmd_odds(args: argparse.Namespace) -> None:
    all_lines: list[BetLine] = []
    with Fetcher() as fetcher:
        if args.from_matches:
            matches = load_json(args.from_matches)
            print(f"Fetching odds for {len(matches)} matches from {args.from_matches}", file=sys.stderr)
            for i, m in enumerate(matches, start=1):
                html = fetcher.get_html(m["match_url"])
                lines = parse_match_odds(html, match_id=m["match_id"], source=m.get("source", ""))
                all_lines.extend(lines)
                print(f"  [{i}/{len(matches)}] match {m['match_id']}: {len(lines)} bet lines", file=sys.stderr)
        else:
            html = fetcher.get_html(f"/{args.match_id}")
            all_lines = parse_match_odds(html, match_id=args.match_id, source=f"match:{args.match_id}")

    print(f"Parsed {len(all_lines)} bet lines total", file=sys.stderr)
    save_json([b.to_dict() for b in all_lines], args.out)
    print(f"Wrote {args.out}", file=sys.stderr)


def cmd_standings(args: argparse.Namespace) -> None:
    with Fetcher() as fetcher:
        html = fetcher.get_html(f"/event/{args.event_id}")

    entries = parse_event_standings(html, event_id=args.event_id)
    print(f"Parsed {len(entries)} standings rows for event {args.event_id}", file=sys.stderr)
    save_json([e.to_dict() for e in entries], args.out)
    print(f"Wrote {args.out}", file=sys.stderr)


def cmd_team(args: argparse.Namespace) -> None:
    with Fetcher() as fetcher:
        html = fetcher.get_html(f"/team/{args.team_id}")

    team = parse_team(html, team_id=args.team_id)
    print(
        f"Parsed team {team.name!r} ({len(team.roster)} players, {len(team.staff)} staff)",
        file=sys.stderr,
    )
    save_json(team.to_dict(), args.out)
    print(f"Wrote {args.out}", file=sys.stderr)


def cmd_closing(args: argparse.Namespace) -> None:
    if args.action in ("poll", "run"):
        summary = closing_lines.poll_once(upcoming_pages=args.upcoming_pages, results_pages=args.results_pages)
        print(
            f"Poll done: {summary['snapshotted']} snapshotted, "
            f"{summary['finalized']} finalized with vig, {summary['missed']} missed pre-match",
            file=sys.stderr,
        )
    if args.action == "backfill":
        summary = closing_lines.backfill_historical(target_total=args.target)
        print(
            f"Backfill done: had {summary['already_had']}, added {summary['added']} "
            f"(scanned {summary['pages_scanned']} results pages)",
            file=sys.stderr,
        )
    if args.action in ("export", "run"):
        n = closing_lines.export_excel()
        print(f"Wrote {n} matches to {closing_lines.XLSX_PATH}", file=sys.stderr)


def cmd_flatten(args: argparse.Namespace) -> None:
    data = load_json(args.input)

    if args.kind == "stats":
        rows = player_stats_to_rows([PlayerStat(**d) for d in data])
    elif args.kind == "standings":
        rows = standings_to_rows([StandingEntry(**d) for d in data])
    elif args.kind == "matches":
        rows = matches_to_rows([MatchSummary(**d) for d in data])
    elif args.kind == "odds":
        rows = odds_to_rows([BetLine(**d) for d in data])
    elif args.kind == "team":
        team = TeamInfo(
            team_id=data["team_id"],
            name=data["name"],
            tag=data["tag"],
            country=data["country"],
            links=data["links"],
            roster=[RosterPlayer(**p) for p in data["roster"]],
            staff=[RosterPlayer(**p) for p in data["staff"]],
        )
        rows = roster_to_rows(team)
    else:
        raise ValueError(f"Unknown kind: {args.kind}")

    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} rows to {args.out}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vlr.gg scraper CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    stats_p = sub.add_parser("stats", help="Scrape a player-stats leaderboard (event or global)")
    stats_p.add_argument("--event-id", type=int, default=None, help="Scrape this event's stats page")
    stats_p.add_argument("--global", dest="use_global", action="store_true", help="Scrape the global /stats page instead")
    stats_p.add_argument("--side", default="all", choices=["all", "t", "ct"])
    stats_p.add_argument("--role", default="all", choices=["all", "controller", "initiator", "sentinel", "duelist"])
    stats_p.add_argument("--agent", default="all")
    stats_p.add_argument("--map-id", dest="map_id", default="all")
    stats_p.add_argument("--min-rounds", dest="min_rounds", default="0")
    stats_p.add_argument("--min-rating", dest="min_rating", default="0", help="Global stats only")
    stats_p.add_argument("--region", default="all", help="Global stats only")
    stats_p.add_argument("--timespan", default="all", help="Global stats only, e.g. 30d/60d/90d/all")
    stats_p.add_argument("--sort", default=None, help="Column to sort by, e.g. rating2, acs, adr")
    stats_p.add_argument("--dir", default="desc", choices=["asc", "desc"])
    stats_p.add_argument("--out", required=True, help="Output raw JSON path")
    stats_p.set_defaults(func=cmd_stats)

    matches_p = sub.add_parser("matches", help="Scrape match schedule/results (global list or one event's match tab)")
    matches_p.add_argument("--event-id", type=int, default=None, help="Scrape this event's match tab instead of the global list")
    matches_p.add_argument("--status", default="results", choices=["upcoming", "results"], help="Global list only: schedule (upcoming) or completed results")
    matches_p.add_argument("--page", type=int, default=1, help="Global list only: starting page number")
    matches_p.add_argument("--pages", type=int, default=1, help="Global list only: how many consecutive pages to fetch")
    matches_p.add_argument("--out", required=True, help="Output raw JSON path")
    matches_p.set_defaults(func=cmd_matches)

    odds_p = sub.add_parser("odds", help="Scrape a match page's betting odds widget")
    odds_p.add_argument("--match-id", type=int, default=None, help="Scrape this single match's odds")
    odds_p.add_argument(
        "--from-matches",
        default=None,
        help="Path to a raw JSON file from the `matches` command; scrapes odds for every match_id/match_url in it",
    )
    odds_p.add_argument("--out", required=True, help="Output raw JSON path")
    odds_p.set_defaults(func=cmd_odds)

    standings_p = sub.add_parser("standings", help="Scrape an event's final standings/bracket placements")
    standings_p.add_argument("--event-id", type=int, required=True)
    standings_p.add_argument("--out", required=True, help="Output raw JSON path")
    standings_p.set_defaults(func=cmd_standings)

    team_p = sub.add_parser("team", help="Scrape a team's info and roster")
    team_p.add_argument("--team-id", type=int, required=True)
    team_p.add_argument("--out", required=True, help="Output raw JSON path")
    team_p.set_defaults(func=cmd_team)

    closing_p = sub.add_parser(
        "closing",
        help="Poll upcoming/live matches for pre-match odds and finalize closing lines + vig once matches complete",
    )
    closing_p.add_argument("action", choices=["poll", "backfill", "export", "run"], help="poll: snapshot+finalize; backfill: extend history; export: rebuild xlsx; run: poll+export")
    closing_p.add_argument("--upcoming-pages", type=int, default=1, help="Pages of /matches (upcoming) to poll")
    closing_p.add_argument("--results-pages", type=int, default=2, help="Pages of /matches/results to check for newly-completed matches")
    closing_p.add_argument("--target", type=int, default=300, help="backfill only: total historical matches to have on record")
    closing_p.set_defaults(func=cmd_closing)

    flatten_p = sub.add_parser("flatten", help="Flatten a raw JSON file into a tabular CSV")
    flatten_p.add_argument("kind", choices=["stats", "standings", "matches", "odds", "team"])
    flatten_p.add_argument("--input", required=True, help="Raw JSON path produced by stats/standings/team")
    flatten_p.add_argument("--out", required=True, help="Output CSV path")
    flatten_p.set_defaults(func=cmd_flatten)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "stats" and args.event_id is None and not args.use_global:
        parser.error("stats requires either --event-id or --global")
    if args.command == "odds" and args.match_id is None and not args.from_matches:
        parser.error("odds requires either --match-id or --from-matches")

    args.func(args)


if __name__ == "__main__":
    main()
