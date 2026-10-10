import sqlite3

import pandas as pd
import pytest

from src import store


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.db"
    with store.connect_rw(path) as conn:
        store.init_db(conn)
        conn.executemany(
            "INSERT INTO t100_route_month VALUES (?,?,?,?,?,?,?,?,?,?)",
            [
                ("ANC", "AK", "FAI", 2025, 1, "pax", 261, 10, 1000, 800),
                ("ANC", "AK", "ORD", 2025, 1, "cargo", 2846, 5, 0, 0),
                ("ANC", "AK", "SEA", 2025, 2, "combi", 1448, 4, 400, 300),
                ("ANC", "AK", "ORD", 2024, 1, "pax", 2846, 9, 900, 900),
            ],
        )
    return path


def test_schema_has_all_tables(db):
    with store.connect_ro(db) as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"meta", "ingest_log", "airports", "t100_route_month", "ontime_airport_month", "census_metro"} <= names


def test_init_db_is_idempotent(db):
    with store.connect_rw(db) as conn:
        store.init_db(conn)
        assert conn.execute("SELECT COUNT(*) FROM t100_route_month").fetchone()[0] == 4


def test_read_only_connection_rejects_writes(db):
    with store.connect_ro(db) as conn, pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM t100_route_month")


def test_read_only_requires_existing_db(tmp_path):
    with pytest.raises(sqlite3.OperationalError):
        store.connect_ro(tmp_path / "missing.db")


def test_route_totals_filters(db):
    with store.connect_ro(db) as conn:
        all_flights = store.route_totals(conn, "ANC", 2025)
        pax = store.route_totals(conn, "ANC", 2025, configs=store.PASSENGER_CONFIGS)
        long_haul = store.route_totals(conn, "ANC", 2025, min_distance=2500)
    assert all_flights["departures"] == 19
    assert pax == {"departures": 14, "seats": 1400, "passengers": 1100}
    assert long_haul["departures"] == 5


def test_route_totals_empty(db):
    with store.connect_ro(db) as conn:
        assert store.route_totals(conn, "XXX", 2025)["departures"] == 0


# --- analysis window and per-airport metrics (tiny in-memory DB) ---------------------------

ALL_2024_2025 = [f"{y}-{m:02d}" for y in (2024, 2025) for m in range(1, 13)]


def _log(conn, source, periods):
    conn.executemany("INSERT INTO ingest_log VALUES (?, ?, 1, 'now')", [(source, p) for p in periods])
    conn.execute("INSERT INTO meta (source, last_period) VALUES (?, ?)", (source, max(periods)))


@pytest.fixture
def mem():
    conn = sqlite3.connect(":memory:")
    store.init_db(conn)
    conn.executemany("INSERT INTO t100_route_month VALUES (?,?,?,?,?,?,?,?,?,?)", [
        ("BOS", "MA", "LAX", 2025, 3, "pax", 2611, 10, 1000, 900),
        ("BOS", "MA", "LAX", 2025, 4, "combi", 2611, 2, 200, 150),
        ("BOS", "MA", "PWM", 2025, 4, "pax", 98, 5, 500, 300),
        ("BOS", "MA", "ANC", 2025, 4, "cargo", 3000, 7, 0, 0),      # cargo: not passenger service
        ("BOS", "MA", "LAX", 2024, 5, "pax", 2611, 9, 900, 800),    # previous window
        ("BOS", "MA", "LAX", 2023, 12, "pax", 2611, 9, 900, 999),   # before the previous window
        ("PWM", "ME", "BOS", 2024, 6, "pax", 98, 3, 300, 200),      # previous window only -> no row
        ("YVR", None, "BOS", 2025, 1, "pax", 2000, 1, 100, 90),     # foreign origin
        ("YYZ", "ON", "BOS", 2025, 1, "pax", 500, 1, 100, 90),      # T-100 gives Canadian provinces a state
    ])
    conn.execute("INSERT INTO airports (iata, name, state, region, cbsa, lat, lon) "
                 "VALUES ('BOS', 'Logan', 'MA', 'new_england', NULL, 0, 0)")
    _log(conn, "t100", ALL_2024_2025)
    yield conn
    conn.close()


def test_window_full_coverage(mem):
    w = store.analysis_window(mem)
    assert (w.start, w.end, w.prev_start, w.prev_end) == ("2025-01", "2025-12", "2024-01", "2024-12")
    assert (w.months, w.prev_months, w.label) == (12, 12, "2025-01..2025-12")


def test_window_ends_at_the_slowest_source(mem):
    _log(mem, "ontime", [f"2025-{m:02d}" for m in range(1, 11)])
    w = store.analysis_window(mem)
    assert (w.start, w.end, w.prev_start, w.prev_end) == ("2024-11", "2025-10", "2023-11", "2024-10")
    assert w.months == 10  # on-time has only 2025-01..2025-10 inside the window


def test_window_requires_data():
    conn = sqlite3.connect(":memory:")
    store.init_db(conn)
    with pytest.raises(ValueError, match="ingest"):
        store.analysis_window(conn)


def test_airport_metrics(mem):
    df = store.airport_metrics(mem, store.analysis_window(mem), long_haul_miles=2500).set_index("iata")
    assert list(df.index) == ["BOS", "YVR", "YYZ"]
    bos = df.loc["BOS"]
    assert (bos["departures"], bos["seats"], bos["passengers"]) == (17, 1700, 1350)
    assert (bos["long_haul_departures"], bos["months_active"], bos["prev_passengers"]) == (12, 2, 800)
    assert (bos["state"], bos["region"], bos["name"]) == ("MA", "new_england", "Logan")
    assert df.loc[["YVR", "YYZ"], "state"].isna().all()  # foreign: never in the peer set
    assert df.loc["YVR", "prev_passengers"] == 0


def test_init_db_adds_municipality_to_an_older_airports_table():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE airports (iata TEXT PRIMARY KEY, name TEXT, state TEXT, region TEXT, "
                 "cbsa TEXT, lat REAL, lon REAL)")
    store.init_db(conn)
    assert "municipality" in {r[1] for r in conn.execute("PRAGMA table_info(airports)")}


def test_find_airports_and_long_haul_counts(mem):
    w = store.analysis_window(mem)
    assert store.find_airports(mem, "name", "log", w) == [
        {"iata": "BOS", "name": "Logan", "state": "MA", "municipality": None, "passengers": 1350}]
    assert store.find_airports(mem, "name", "%", w) == []  # wildcards are literal
    every = store.long_haul_counts(mem, "BOS", w, [2500, 3000])
    pax = store.long_haul_counts(mem, "BOS", w, [2500, 3000], store.PASSENGER_CONFIGS)
    assert every == {"departures": 24, "long_haul": {2500: 19, 3000: 7}}
    assert pax == {"departures": 17, "long_haul": {2500: 12, 3000: 0}}
