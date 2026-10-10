from src import ingest
from tests.conftest import T100_HEADER, t100_row


def rows(conn):
    return conn.execute(
        "SELECT origin, origin_state, dest, year, month, config, distance, departures, seats, passengers "
        "FROM t100_route_month ORDER BY month, origin, dest, config"
    ).fetchall()


def test_t100_zero_departures_dropped(conn, t100_dir):
    ingest.ingest_t100(conn, t100_dir)
    assert not [r for r in rows(conn) if r[2] == "ANC" and r[0] == "ANC"]


def test_t100_config_mapping(conn, t100_dir):
    ingest.ingest_t100(conn, t100_dir)
    configs = {(r[2], r[5]) for r in rows(conn)}
    assert configs == {("FAI", "pax"), ("ORD", "cargo"), ("SEA", "combi"), ("SEA", "unknown"), ("ANC", "pax")}


def test_t100_aggregation_sums(conn, t100_dir):
    loaded = ingest.ingest_t100(conn, t100_dir)
    fai = [r for r in rows(conn) if r[2] == "FAI"]
    assert fai == [("ANC", "AK", "FAI", 2025, 1, "pax", 261, 15, 1500, 1250)]
    assert loaded == 5


def test_t100_foreign_origin_has_null_state(conn, t100_dir):
    ingest.ingest_t100(conn, t100_dir)
    assert [r[1] for r in rows(conn) if r[0] == "YVR"] == [None]


def test_t100_rerun_is_idempotent(conn, t100_dir):
    assert ingest.ingest_t100(conn, t100_dir) == 5
    before = rows(conn)
    assert ingest.ingest_t100(conn, t100_dir) == 0
    assert rows(conn) == before
    periods = conn.execute("SELECT period FROM ingest_log WHERE source='t100' ORDER BY period").fetchall()
    assert periods == [("2025-01",), ("2025-02",)]
    assert conn.execute("SELECT last_period FROM meta WHERE source='t100'").fetchone() == ("2025-02",)


def test_t100_new_month_in_new_file_is_added(conn, t100_dir):
    ingest.ingest_t100(conn, t100_dir)
    (t100_dir / "t100_2025_03.csv").write_text(
        T100_HEADER
        + t100_row("4.00", "400.00", "300.00", "261.00", "ANC", "FAI", "1", 2025, 3)
        # month 1 again in another file: already loaded, skipped
        + t100_row("9.00", "900.00", "900.00", "261.00", "ANC", "FAI", "1", 2025, 1),
        encoding="utf-8",
    )
    assert ingest.ingest_t100(conn, t100_dir) == 1
    assert conn.execute("SELECT SUM(departures) FROM t100_route_month WHERE month=1 AND dest='FAI'").fetchone() == (15,)


