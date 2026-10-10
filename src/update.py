"""`uv run python -m src.ingest --update [--through YYYY-MM] [--dry-run] [--yes]` (P4.6). Opt-in.

Brings the two monthly BTS sources (On-Time, T-100) up to date, one month at a time:
1. The `--check` logic (src/upstream.py) gives the upstream latest per source; the plan is every
   month after the loaded one up to `--through` (default: the upstream latest) that is not loaded.
2. The plan (months, rough size and time) is printed; nothing is downloaded with `--dry-run`, and
   without `--yes` the user confirms (a non-interactive run without `--yes` aborts).
3. Per month: download into a temp folder outside the project (src/download.py), validate, load
   through the existing aggregation in src/ingest.py and commit. The first failure stops the run;
   earlier months stay loaded and the DB stays consistent (one transaction per month).
A month already loaded is never reloaded, so BTS revisions of loaded months are not picked up.
The chat never calls this: only this command downloads data.
"""
from __future__ import annotations

import re
import sqlite3
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

from src import census, download, ingest, store, upstream
from src.config import ROOT, ScoringConfig, Settings, SourcesConfig

SOURCES = (ingest.ONTIME_SOURCE, ingest.T100_SOURCE)   # the monthly sources that set the window
PERIOD = re.compile(r"\d{4}-(0[1-9]|1[0-2])")


def period_arg(text: str) -> str:
    """argparse type for --through: 'YYYY-MM'."""
    if not PERIOD.fullmatch(text):
        raise ValueError(f"{text!r} is not a month like 2026-01")
    return text


def months_between(after: str, through: str) -> list[str]:
    """Months after `after` up to and including `through` ('YYYY-MM')."""
    months, period = [], store._shift(after, 1)
    while period <= through:
        months.append(period)
        period = store._shift(period, 1)
    return months


@dataclass(frozen=True)
class SourcePlan:
    source: str
    loaded: str | None
    upstream: str | None
    months: tuple[str, ...]
    note: str = ""


def plan_source(conn: sqlite3.Connection, source: str, loaded: str | None, upstream_latest: str | None,
                through: str | None) -> SourcePlan:
    """Missing months of one source up to `through` (capped at the upstream latest when known)."""
    if not loaded:
        return SourcePlan(source, loaded, upstream_latest, (), "nothing loaded yet: run `python -m src.ingest` first")
    note, target = "", upstream_latest
    if through and upstream_latest and through > upstream_latest:
        note = f"upstream has only {upstream_latest}, not {through}"
    elif through:
        target = through
    if target is None:
        return SourcePlan(source, loaded, None, (), "upstream latest unknown: nothing planned (pass --through to try)")
    if through and not upstream_latest:
        note = "upstream latest unknown: trying up to --through"
    done = store.loaded_periods(conn, source)
    months = tuple(m for m in months_between(loaded, target) if m not in done)
    return SourcePlan(source, loaded, upstream_latest, months, note)


def estimate(plans: list[SourcePlan], src: SourcesConfig, delay_s: float) -> tuple[float, float]:
    """Rough download size (MB) and time (seconds) of the plan, from config/sources.yaml."""
    size = {ingest.ONTIME_SOURCE: src.ontime_month_mb, ingest.T100_SOURCE: src.t100_month_mb}
    mb = sum(size[p.source] * len(p.months) for p in plans)
    requests_made = sum(len(p.months) * (2 if p.source == ingest.T100_SOURCE else 1) for p in plans)
    return mb, mb * 1000 / src.assumed_kb_per_s + requests_made * delay_s


def plan_lines(plans: list[SourcePlan], mb: float, seconds: float, src: SourcesConfig) -> list[str]:
    lines = []
    for p in plans:
        months = f"{len(p.months)} months: {p.months[0]}..{p.months[-1]}" if p.months else "nothing to load"
        lines.append(f"{p.source:<7} loaded {p.loaded or 'none'}, upstream {p.upstream or 'unknown'}: {months}"
                     + (f" ({p.note})" if p.note else ""))
    if any(p.months for p in plans):
        lines.append(f"download: about {mb:,.0f} MB, about {seconds / 60:,.0f} min at {src.assumed_kb_per_s:,.0f} KB/s "
                     "(BTS is slow; On-Time is most of it)")
    return lines


def confirm(yes: bool, interactive: bool, ask: Callable[[str], str]) -> str | None:
    """None to go ahead, else the reason to stop (printed)."""
    if yes:
        return None
    if not interactive:
        return "Not an interactive terminal: rerun with --yes to download, or --dry-run to only see the plan."
    answer = ask("Download and load these months? [y/N] ").strip().lower()
    return None if answer in ("y", "yes") else "Aborted: nothing downloaded."


def load_month(conn: sqlite3.Connection, client: download.PoliteClient, settings: Settings, src: SourcesConfig,
               source: str, period: str, work: Path) -> str:
    """Download, validate and load one month of one source in one transaction; returns a summary line.
    Raises (download.DownloadError, ValueError) with nothing written for this month."""
    started = time.perf_counter()
    try:
        if source == ingest.ONTIME_SOURCE:
            required = list(ingest.ontime_source_columns(src.ontime_prezip_columns))
            path = download.fetch_ontime(client, settings.ontime_prezip_url, period, work, required)
            size = path.stat().st_size
            month = ingest.load_ontime_file(conn, path, rename=src.ontime_prezip_columns, expect_period=period)
            if month is None:
                raise ValueError(f"this file is already loaded under another month ({path.name})")
            summary = f"{month.flights:,} flights, {len(month.table)} airports"
        else:
            path = download.fetch_t100(client, settings.t100_page_url, src.t100_form, src.t100_fields, period, work,
                                       src.t100_zip_csv, ingest.T100_COLUMNS)
            size = path.stat().st_size
            rows = ingest.load_t100_file(conn, path, expect_period=period)
            pax = rows.loc[rows["config"].isin(store.PASSENGER_CONFIGS), "passengers"].sum()
            summary = f"{len(rows):,} rows, {pax / 1e6:.1f}M passenger-service passengers"
        ingest.update_meta(conn, source)
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        for leftover in work.iterdir():
            leftover.unlink()
    return f"{source:<7} {period}: {summary} ({size / 1e6:.1f} MB file, {time.perf_counter() - started:.0f} s)"


def run(conn: sqlite3.Connection, settings: Settings, cfg: ScoringConfig, src: SourcesConfig, *,
        through: str | None = None, dry_run: bool = False, yes: bool = False, session=None,
        interactive: bool | None = None, ask: Callable[[str], str] = input, sleep=time.sleep,
        geocode=ingest.geocode_peer_airports, today: date | None = None, out=print) -> int:
    """The whole update; returns the exit code (0 done or nothing to do, 1 failed or declined,
    2 not interactive without --yes)."""
    session = session or download.session(settings.http_user_agent)
    checked = {r.source: r for r in upstream.run_check(conn, session, settings, cfg, today)}
    conn.commit()   # the check result (meta.upstream_latest) is kept whatever happens next
    plans = [plan_source(conn, s, checked[s].loaded if s in checked else None,
                         checked[s].upstream if s in checked else None, through) for s in SOURCES]
    mb, seconds = estimate(plans, src, settings.download_delay_s)
    for line in plan_lines(plans, mb, seconds, src):
        out(line)
    todo = sorted((m, s) for p in plans for m in p.months for s in [p.source])   # month order, On-Time first
    if not todo:
        out("Nothing to update.")
        return 0
    if dry_run:
        out("Dry run: nothing downloaded.")
        return 0
    stop = confirm(yes, sys.stdin.isatty() if interactive is None else interactive, ask)
    if stop:
        out(stop)
        return 2 if stop.startswith("Not an interactive") else 1

    client = download.PoliteClient(session, settings.download_timeout_s, settings.download_delay_s,
                                   settings.download_retries, int(src.max_download_mb * 1e6), sleep)
    loaded: list[tuple[str, str]] = []
    code = 0
    with tempfile.TemporaryDirectory(prefix="airport-update-") as tmp:
        work = Path(tmp)
        if work.resolve().is_relative_to(ROOT):
            raise RuntimeError(f"temp folder {work} is inside the project; set TMPDIR elsewhere")
        for period, source in todo:
            try:
                out(load_month(conn, client, settings, src, source, period, work))
                loaded.append((source, period))
            except (download.DownloadError, ValueError, OSError) as exc:
                out(f"{source:<7} {period}: FAILED, nothing written for this month: {exc}")
                out("Stopped. Months loaded before this one are kept; rerun --update to continue.")
                code = 1
                break
    report(conn, settings, cfg, src, loaded, session, geocode, out)
    return code


def report(conn: sqlite3.Connection, settings: Settings, cfg: ScoringConfig, src: SourcesConfig,
           loaded: list[tuple[str, str]], session, geocode, out) -> None:
    """After the run: row checks, CBSAs for new peer airports, loaded periods, window, freshness."""
    ontime_months = [p for s, p in loaded if s == ingest.ONTIME_SOURCE]
    for line in ingest.ontime_row_check(conn, src.ontime_expected_rows, ontime_months):
        out(line)
    if loaded:   # a new window can admit new peer airports, which need a CBSA (cached, so usually no request)
        geo = geocode(conn, census.session(), settings, cfg)
        conn.commit()
        out(f"geocoder: {len(geo.matched)} of {geo.total} peer-set airports matched to a CBSA "
            f"({geo.requested} requests" + (f", {len(geo.failed)} failed, retried next run" if geo.failed else "") + ")")
    rows = store.meta_rows(conn)
    out("loaded: " + ", ".join(f"{r['source']} {r['last_period']}" for r in rows if r["source"] in SOURCES))
    window = store.analysis_window(conn)
    out(f"analysis window: {window.label} ({window.months} of 12 months), growth vs "
        f"{window.prev_start}..{window.prev_end}")
    out(f"freshness: {store.overall_freshness(rows)}")
