"""SQLite store: schema and small query helpers.

Chat-time code opens the DB read-only (`connect_ro`); only `src.ingest` writes (`connect_rw`).
Schema and definitions: docs/plan.md.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

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
    lon REAL
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
