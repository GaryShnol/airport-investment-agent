"""SQLite store: schema and small query helpers.

Chat-time code opens the DB read-only (`connect_ro`); only `src.ingest` writes (`connect_rw`).
Schema and definitions: docs/plan.md.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

# Passenger service = AIRCRAFT_CONFIG 1 (pax) or 3 (combi); see Definitions in docs/plan.md.
PASSENGER_CONFIGS: tuple[str, ...] = ("pax", "combi")

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    source TEXT PRIMARY KEY,
    last_period TEXT,          -- latest loaded month, 'YYYY-MM'
    upstream_latest TEXT,      -- latest month published upstream (from ingest --check)
    checked_at TEXT,
    fetched_at TEXT
);
-- One row per loaded (source, month): makes ingest idempotent.
CREATE TABLE IF NOT EXISTS ingest_log (
    source TEXT NOT NULL,
    period TEXT NOT NULL,      -- 'YYYY-MM'
    rows INTEGER NOT NULL,
    loaded_at TEXT NOT NULL,
    PRIMARY KEY (source, period)
);
CREATE TABLE IF NOT EXISTS airports (
    iata TEXT PRIMARY KEY,
    name TEXT,
    state TEXT,
    region TEXT,
    cbsa TEXT,
    lat REAL,
    lon REAL,
    municipality TEXT          -- city served (OurAirports), used to resolve "Santa Ana" -> SNA
);
CREATE TABLE IF NOT EXISTS t100_route_month (
    origin TEXT NOT NULL,
    origin_state TEXT,         -- ORIGIN_STATE_ABR, null for foreign origins (peer-set filter)
    dest TEXT NOT NULL,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    config TEXT NOT NULL CHECK (config IN ('pax', 'cargo', 'combi', 'unknown')),
    distance INTEGER NOT NULL,
    departures INTEGER NOT NULL,
    seats INTEGER NOT NULL,
    passengers INTEGER NOT NULL,
    PRIMARY KEY (origin, dest, year, month, config, distance)
);
CREATE INDEX IF NOT EXISTS ix_t100_period ON t100_route_month (year, month);
CREATE TABLE IF NOT EXISTS ontime_airport_month (
    airport TEXT NOT NULL,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    dep_flights INTEGER, dep_del15 INTEGER, dep_cancelled INTEGER,
    taxi_out_sum REAL, taxi_out_n INTEGER,
    arr_flights INTEGER, arr_del15 INTEGER,
    carrier_min REAL, weather_min REAL, nas_min REAL, security_min REAL, late_aircraft_min REAL,
    PRIMARY KEY (airport, year, month)
);
CREATE TABLE IF NOT EXISTS census_metro (
    cbsa TEXT NOT NULL,
    year INTEGER NOT NULL,
    population INTEGER,
    fetched_at TEXT,
    PRIMARY KEY (cbsa, year)
);
"""


@contextmanager
def connect_rw(path: Path) -> Iterator[sqlite3.Connection]:
    """Read-write connection for ingest. Commits on success, rolls back on error."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def connect_ro(path: Path) -> sqlite3.Connection:
    """Read-only connection (`mode=ro`); fails if the DB does not exist instead of creating it."""
    return sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    # Columns added after a DB was first created (CREATE TABLE IF NOT EXISTS does not add them).
    columns = {r[1] for r in conn.execute("PRAGMA table_info(airports)")}
    if "municipality" not in columns:
        conn.execute("ALTER TABLE airports ADD COLUMN municipality TEXT")


def loaded_periods(conn: sqlite3.Connection, source: str) -> set[str]:
    return {r[0] for r in conn.execute("SELECT period FROM ingest_log WHERE source = ?", (source,))}


def route_totals(
    conn: sqlite3.Connection,
    origin: str,
    year: int,
    configs: Iterable[str] | None = None,
    min_distance: int = 0,
) -> dict[str, int]:
    """Departures, seats and passengers out of `origin` in `year` (all configs unless given)."""
    sql = (
        "SELECT COALESCE(SUM(departures), 0), COALESCE(SUM(seats), 0), COALESCE(SUM(passengers), 0) "
        "FROM t100_route_month WHERE origin = ? AND year = ? AND distance >= ?"
    )
    params: list = [origin, year, min_distance]
    if configs is not None:
        configs = tuple(configs)
        sql += f" AND config IN ({','.join('?' * len(configs))})"
        params += configs
    departures, seats, passengers = conn.execute(sql, params).fetchone()
    return {"departures": departures, "seats": seats, "passengers": passengers}


def unmatched_origins(conn: sqlite3.Connection) -> list[str]:
    """T-100 origins with no row in `airports` (shown by ingest; such airports lack name and region)."""
    rows = conn.execute(
        "SELECT DISTINCT origin FROM t100_route_month "
        "WHERE origin NOT IN (SELECT iata FROM airports) ORDER BY origin"
    )
    return [r[0] for r in rows]


WINDOW_MONTHS = 12


def _shift(period: str, months: int) -> str:
    """'YYYY-MM' moved by `months` (negative goes back)."""
    year, month = map(int, period.split("-"))
    index = year * 12 + month - 1 + months
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def _key(period: str) -> int:
    """'YYYY-MM' -> year * 100 + month, comparable with the table columns."""
    year, month = map(int, period.split("-"))
    return year * 100 + month


@dataclass(frozen=True)
class Window:
    """Analysis window (trailing 12 months) and the previous 12 months used for growth.

    `months` / `prev_months` count months loaded inside each window, for every loaded source
    (the minimum across sources); fewer than 12 means partial coverage.
    """
    start: str
    end: str
    prev_start: str
    prev_end: str
    months: int
    prev_months: int

    @property
    def label(self) -> str:
        return f"{self.start}..{self.end}"


def analysis_window(conn: sqlite3.Connection) -> Window:
    """Trailing 12 months ending at the minimum `last_period` over loaded sources (docs/plan.md)."""
    rows = conn.execute("SELECT source, last_period FROM meta WHERE last_period IS NOT NULL").fetchall()
    if not rows:
        raise ValueError("No data loaded: run `uv run python -m src.ingest` first")
    end = min(last for _, last in rows)
    start = _shift(end, -(WINDOW_MONTHS - 1))
    prev_end, prev_start = _shift(start, -1), _shift(start, -WINDOW_MONTHS)

    def coverage(lo: str, hi: str) -> int:
        return min(
            conn.execute(
                "SELECT COUNT(*) FROM ingest_log WHERE source = ? AND period BETWEEN ? AND ?", (source, lo, hi)
            ).fetchone()[0]
            for source, _ in rows
        )

    return Window(start, end, prev_start, prev_end, coverage(start, end), coverage(prev_start, prev_end))


def airport_metrics(conn: sqlite3.Connection, window: Window, long_haul_miles: int) -> pd.DataFrame:
    """One row per origin airport with passenger service in the window.

    Columns: iata, name, state, region, departures, seats, passengers, long_haul_departures
    (distance >= long_haul_miles), months_active, prev_passengers (previous 12 months; 0 if none).
    All counts are passenger service only (PASSENGER_CONFIGS). Name, state and region come from
    `airports`: state is set only for US states and territories, null for foreign airports and
    for codes without a match. T-100 ORIGIN_STATE_ABR is not used here because it also holds
    Canadian province codes (ON, BC, ...) and TT for the Pacific territories.
    """
    config_params = {f"c{i}": c for i, c in enumerate(PASSENGER_CONFIGS)}
    configs = ",".join(f":{name}" for name in config_params)
    sql = f"""
        WITH t AS (
            SELECT *, year * 100 + month AS period,
                   (year * 100 + month) BETWEEN :start AND :end AS cur
            FROM t100_route_month
            WHERE config IN ({configs}) AND (year * 100 + month) BETWEEN :prev_start AND :end
        )
        SELECT t.origin AS iata, a.name, a.state, a.region,
               SUM(CASE WHEN cur THEN departures ELSE 0 END) AS departures,
               SUM(CASE WHEN cur THEN seats ELSE 0 END) AS seats,
               SUM(CASE WHEN cur THEN passengers ELSE 0 END) AS passengers,
               SUM(CASE WHEN cur AND distance >= :lh THEN departures ELSE 0 END) AS long_haul_departures,
               COUNT(DISTINCT CASE WHEN cur THEN period END) AS months_active,
               SUM(CASE WHEN cur THEN 0 ELSE passengers END) AS prev_passengers
        FROM t LEFT JOIN airports a ON a.iata = t.origin
        GROUP BY t.origin
        HAVING SUM(CASE WHEN cur THEN departures ELSE 0 END) > 0
        ORDER BY t.origin
    """
    params = {"start": _key(window.start), "end": _key(window.end), "prev_start": _key(window.prev_start),
              "lh": long_haul_miles, **config_params}
    return pd.read_sql_query(sql, conn, params=params)


def meta_rows(conn: sqlite3.Connection) -> list[dict]:
    """Freshness summary, one dict per loaded source."""
    rows = conn.execute("SELECT source, last_period, upstream_latest, checked_at FROM meta ORDER BY source")
    return [dict(zip(("source", "last_period", "upstream_latest", "checked_at"), r)) for r in rows]


_SEARCH_COLUMNS = ("iata", "name", "municipality")


def find_airports(conn: sqlite3.Connection, column: str, text: str, window: Window) -> list[dict]:
    """Airports whose `column` equals (iata) or contains (name, municipality) `text`, case-insensitive.

    Each row carries passenger-service passengers in the window (0 if none), largest first, so
    callers can ignore airports without service (heliports, closed fields).
    """
    if column not in _SEARCH_COLUMNS:
        raise ValueError(f"column must be one of {_SEARCH_COLUMNS}")
    if column == "iata":
        where, value = "a.iata = ?", text.upper()
    else:
        escaped = text.replace("!", "!!").replace("%", "!%").replace("_", "!_")
        where, value = f"a.{column} LIKE ? ESCAPE '!'", f"%{escaped}%"
    configs = ",".join("?" * len(PASSENGER_CONFIGS))
    sql = f"""
        SELECT a.iata, a.name, a.state, a.municipality,
               (SELECT COALESCE(SUM(t.passengers), 0) FROM t100_route_month t
                WHERE t.origin = a.iata AND t.config IN ({configs})
                  AND (t.year * 100 + t.month) BETWEEN ? AND ?) AS passengers
        FROM airports a WHERE {where}
        ORDER BY passengers DESC, a.iata
    """
    params = [*PASSENGER_CONFIGS, _key(window.start), _key(window.end), value]
    cols = ("iata", "name", "state", "municipality", "passengers")
    return [dict(zip(cols, r)) for r in conn.execute(sql, params)]


def served_airports(conn: sqlite3.Connection, window: Window) -> list[dict]:
    """Airports with passenger service in the window (iata, name, state, municipality, passengers),
    largest first. Used to suggest names when resolve_airport finds no match."""
    configs = ",".join("?" * len(PASSENGER_CONFIGS))
    sql = f"""
        SELECT a.iata, a.name, a.state, a.municipality, SUM(t.passengers) AS passengers
        FROM t100_route_month t JOIN airports a ON a.iata = t.origin
        WHERE t.config IN ({configs}) AND (t.year * 100 + t.month) BETWEEN ? AND ?
        GROUP BY a.iata HAVING SUM(t.passengers) > 0
        ORDER BY passengers DESC, a.iata
    """
    cols = ("iata", "name", "state", "municipality", "passengers")
    params = [*PASSENGER_CONFIGS, _key(window.start), _key(window.end)]
    return [dict(zip(cols, r)) for r in conn.execute(sql, params)]


def long_haul_counts(
    conn: sqlite3.Connection,
    origin: str,
    window: Window,
    thresholds: Iterable[int],
    configs: Iterable[str] | None = None,
) -> dict:
    """Departures out of `origin` in the window, and departures at or above each threshold (miles).

    All configs unless `configs` is given (pass PASSENGER_CONFIGS for passenger service).
    Returns {"departures": n, "long_haul": {threshold: n}}.
    """
    thresholds = tuple(thresholds)
    cases = ", ".join("COALESCE(SUM(CASE WHEN distance >= ? THEN departures ELSE 0 END), 0)" for _ in thresholds)
    sql = (f"SELECT COALESCE(SUM(departures), 0){', ' + cases if cases else ''} FROM t100_route_month "
           "WHERE origin = ? AND (year * 100 + month) BETWEEN ? AND ?")
    params: list = [*thresholds, origin.upper(), _key(window.start), _key(window.end)]
    if configs is not None:
        configs = tuple(configs)
        sql += f" AND config IN ({','.join('?' * len(configs))})"
        params += configs
    total, *counts = conn.execute(sql, params).fetchone()
    return {"departures": total, "long_haul": dict(zip(thresholds, counts))}
