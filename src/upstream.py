"""Upstream freshness check for `uv run python -m src.ingest --check` (never called at chat time).

For each loaded source, the latest period published upstream:
- On-Time: HEAD the PREZIP URL month by month after the loaded month; 200 with a ZIP = published,
  404 = not published yet (the next months are not tried).
- T-100: the "Latest Available Data: <Month YYYY>" label on the BTS download page.
- Census: the newest vintage of the configured dataset whose variable metadata exists.
Any network failure or unexpected answer gives None ("unknown"), never an error. Results go to
`meta.upstream_latest` and `meta.checked_at`; freshness is derived from them in src/store.py.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone

import requests

from src import store
from src.config import ScoringConfig, Settings

MONTHS = {name: i for i, name in enumerate(
    ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"), start=1)}
T100_LATEST = re.compile(r"Latest Available Data:?\s*(?:<[^>]+>\s*)*([A-Za-z]+)\s+(\d{4})", re.IGNORECASE)
ZIP_TYPES = ("zip", "octet-stream")


def _next_month(period: str) -> str:
    year, month = map(int, period.split("-"))
    return f"{year + month // 12:04d}-{month % 12 + 1:02d}"


def check_ontime(session, url_template: str, last_period: str, today: date, timeout: float) -> str | None:
    """Latest On-Time month published: months after `last_period` up to `today` are tried in order
    until one returns 404. A 200 that is not a ZIP, another status or a network error -> None."""
    latest, period = last_period, _next_month(last_period)
    while period <= f"{today.year:04d}-{today.month:02d}":
        year, month = map(int, period.split("-"))
        try:
            response = session.head(url_template.format(year=year, month=month), timeout=timeout,
                                    allow_redirects=True)
        except requests.RequestException:
            return None
        if response.status_code == 404:
            break
        content_type = response.headers.get("Content-Type", "").lower()
        if response.status_code != 200 or not any(t in content_type for t in ZIP_TYPES):
            return None
        latest, period = period, _next_month(period)
    return latest


def parse_t100_latest(html: str) -> str | None:
    """'Latest Available Data: July 2026' -> '2026-07'; None when the label is missing or unreadable."""
    match = T100_LATEST.search(html)
    if not match or match.group(1).lower() not in MONTHS:
        return None
    return f"{int(match.group(2)):04d}-{MONTHS[match.group(1).lower()]:02d}"


def check_t100(session, page_url: str, timeout: float) -> str | None:
    """Latest T-100 month from the download page label; None if the page cannot be read or parsed."""
    try:
        response = session.get(page_url, timeout=timeout)
    except requests.RequestException:
        return None
    return parse_t100_latest(response.text) if response.status_code == 200 else None


def check_census(session, api_url: str, dataset: str, variable: str, vintage: int, today: date,
                 timeout: float) -> str | None:
    """Newest vintage published: vintages after `vintage` up to the current year are tried until one
    has no metadata (404). The metadata endpoint needs no key. Errors -> None."""
    latest = vintage
    for year in range(vintage + 1, today.year + 1):
        try:
            response = session.get(f"{api_url}/{year}/{dataset}/variables/{variable}.json", timeout=timeout)
        except requests.RequestException:
            return None
        if response.status_code == 404:
            break
        if response.status_code != 200 or "json" not in response.headers.get("Content-Type", "").lower():
            return None
        latest = year
    return str(latest)


@dataclass(frozen=True)
class CheckRow:
    source: str
    loaded: str
    upstream: str | None
    freshness: str


def run_check(conn: sqlite3.Connection, session, settings: Settings, cfg: ScoringConfig,
              today: date | None = None) -> list[CheckRow]:
    """Check every loaded source and store `upstream_latest` (None = unknown) and `checked_at`."""
    today = today or datetime.now(timezone.utc).date()
    checked_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for meta in store.meta_rows(conn):
        source, loaded = meta["source"], meta["last_period"]
        if not loaded:
            continue
        if source == store.ONTIME_SOURCE:
            upstream = check_ontime(session, settings.ontime_prezip_url, loaded, today, settings.http_timeout_s)
        elif source == "t100":
            upstream = check_t100(session, settings.t100_page_url, settings.http_timeout_s)
        elif source == store.CENSUS_SOURCE:
            upstream = check_census(session, settings.census_api_url, cfg.census_dataset, cfg.census_variable,
                                    int(loaded), today, settings.http_timeout_s)
        else:
            upstream = None
        conn.execute("UPDATE meta SET upstream_latest = ?, checked_at = ? WHERE source = ?",
                     (upstream, checked_at, source))
        rows.append(CheckRow(source, loaded, upstream,
                             store.source_freshness({**meta, "upstream_latest": upstream, "checked_at": checked_at})))
    return rows
