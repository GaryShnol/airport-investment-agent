import sqlite3

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
