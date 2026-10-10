"""On-Time ingest on tiny synthetic ZIPs (real BTS header, never data/raw)."""
import zipfile

import pytest

from src import ingest, store

# Header of the BTS TranStats download (T_ONTIME_REPORTING.csv), in file order.
HEADER = ("YEAR,MONTH,FL_DATE,OP_UNIQUE_CARRIER,ORIGIN,ORIGIN_STATE_ABR,DEST,DEST_STATE_ABR,CRS_DEP_TIME,"
          "DEP_DELAY,DEP_DEL15,TAXI_OUT,TAXI_IN,ARR_DELAY,ARR_DEL15,CANCELLED,CANCELLATION_CODE,DIVERTED,"
          "AIR_TIME,FLIGHTS,DISTANCE,CARRIER_DELAY,WEATHER_DELAY,NAS_DELAY,SECURITY_DELAY,LATE_AIRCRAFT_DELAY")


def flight(origin, dest, dep_del15="0.00", taxi_out="10.00", arr_del15="0.00", cancelled="0.00",
           diverted="0.00", causes=("", "", "", "", ""), month=1):
    """One CSV row in the BTS format (numbers as strings like '0.00', empty = not reported)."""
    return ",".join([
        "2025", str(month), f"{month}/1/2025 12:00:00 AM", "AA", origin, "XX", dest, "YY", "0500",
        "", dep_del15, taxi_out, "", "", arr_del15, cancelled, "B" if cancelled == "1.00" else "", diverted,
        "", "1.00", "500.00", *causes,
    ])


FLIGHTS = [
    # SNA -> LAX, delayed on both sides; causes recorded for the arrival at LAX
    flight("SNA", "LAX", dep_del15="1.00", taxi_out="20.00", arr_del15="1.00",
           causes=("10.00", "0.00", "30.00", "0.00", "5.00")),
    flight("SNA", "LAX", taxi_out="14.00"),
    # LAX -> SNA on time
    flight("LAX", "SNA", taxi_out="30.00"),
    # cancelled at SNA: no taxi-out, no delay, not an arrival at LAX
    flight("SNA", "LAX", dep_del15="", taxi_out="", arr_del15="", cancelled="1.00"),
    # diverted LAX -> SNA: departed (counts on the LAX side), never arrived at SNA
    flight("LAX", "SNA", dep_del15="1.00", taxi_out="16.00", arr_del15="", diverted="1.00"),
]


def write_zip(path, rows, newline="\r\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("T_ONTIME_REPORTING.csv", newline.join([HEADER, *rows]) + newline)
    return path


@pytest.fixture
def ontime_dir(tmp_path):
    d = tmp_path / "raw" / "ontime"
    write_zip(d / "_zips" / "ontime_2025_01.zip", FLIGHTS)
    return d


def table(conn):
    cols = ingest.ONTIME_TABLE_COLUMNS
    return {r[0]: dict(zip(cols, r)) for r in conn.execute(
        f"SELECT {','.join(cols)} FROM ontime_airport_month ORDER BY airport, year, month")}


def test_sides_departure_at_origin_arrival_and_causes_at_destination(conn, ontime_dir):
    ingest.ingest_ontime(conn, ontime_dir)
    t = table(conn)
    sna, lax = t["SNA"], t["LAX"]
    # departure side: flights leaving the airport
    assert (sna["dep_flights"], sna["dep_cancelled"], sna["dep_del15"], sna["dep_del15_n"]) == (3, 1, 1, 2)
    assert (sna["taxi_out_sum"], sna["taxi_out_n"]) == (34.0, 2)
    assert (lax["dep_flights"], lax["dep_del15"], lax["dep_del15_n"], lax["taxi_out_sum"]) == (2, 1, 2, 46.0)
    # arrival side: arrivals operated and cause minutes at the destination only
    assert (lax["arr_flights"], lax["arr_del15"], lax["nas_min"], lax["carrier_min"]) == (2, 1, 30.0, 10.0)
    assert lax["late_aircraft_min"] == 5.0
    assert (sna["arr_flights"], sna["arr_del15"], sna["nas_min"]) == (1, 0, 0.0)  # diverted never arrived


def test_cancelled_flights_are_scheduled_departures_but_skip_empty_fields(tmp_path):
    month = ingest.read_ontime(write_zip(tmp_path / "ontime_2025_01.zip", FLIGHTS))
    assert (month.flights, month.cancelled, month.diverted) == (5, 1, 1)
    assert (month.cancelled_with_dep_delay, month.cancelled_with_taxi_out) == (0, 0)


def test_cancelled_flight_that_reports_a_departure_delay_is_kept(tmp_path):
    rows = [flight("SNA", "LAX", dep_del15="1.00", taxi_out="", arr_del15="", cancelled="1.00")]
    month = ingest.read_ontime(write_zip(tmp_path / "ontime_2025_01.zip", rows))
    sna = month.table.set_index("airport").loc["SNA"]
    assert (sna["dep_del15"], sna["dep_del15_n"], sna["dep_cancelled"]) == (1, 1, 1)
    assert month.cancelled_with_dep_delay == 1


def test_streaming_in_small_chunks_gives_the_same_result(tmp_path):
    path = write_zip(tmp_path / "ontime_2025_01.zip", FLIGHTS)
    whole = ingest.read_ontime(path).table
    chunked = ingest.read_ontime(path, chunk_rows=2).table
    assert chunked.equals(whole)


def test_rerun_is_idempotent_and_does_not_reread(conn, ontime_dir, monkeypatch):
    assert [m.period for m in ingest.ingest_ontime(conn, ontime_dir)] == ["2025-01"]
    before = table(conn)
    monkeypatch.setattr(ingest, "read_ontime", lambda *a, **k: pytest.fail("file read again"))
    assert ingest.ingest_ontime(conn, ontime_dir) == []
    assert table(conn) == before
    log = conn.execute("SELECT period, rows, source_rows FROM ingest_log WHERE source='ontime'").fetchall()
    assert log == [("2025-01", 2, 5)]
    assert conn.execute("SELECT last_period FROM meta WHERE source='ontime'").fetchone() == ("2025-01",)


def test_changed_file_reloads_only_its_month(conn, ontime_dir):
    feb = write_zip(ontime_dir / "_zips" / "ontime_2025_02.zip", [flight("SNA", "LAX", month=2)])
    ingest.ingest_ontime(conn, ontime_dir)
    feb_loaded = conn.execute("SELECT loaded_at, file_hash FROM ingest_log WHERE period='2025-02'").fetchone()
    # January is republished with one more flight
    write_zip(ontime_dir / "_zips" / "ontime_2025_01.zip", [*FLIGHTS, flight("SNA", "LAX")])
    assert [m.period for m in ingest.ingest_ontime(conn, ontime_dir)] == ["2025-01"]
    assert conn.execute("SELECT dep_flights FROM ontime_airport_month WHERE airport='SNA' AND month=1"
                        ).fetchone() == (4,)
    assert conn.execute("SELECT COUNT(*) FROM ingest_log WHERE source='ontime'").fetchone() == (2,)
    assert conn.execute("SELECT loaded_at, file_hash FROM ingest_log WHERE period='2025-02'").fetchone() == feb_loaded
    assert feb.exists()


def test_file_with_several_months_is_rejected(tmp_path):
    path = write_zip(tmp_path / "ontime_2025_01.zip", [flight("SNA", "LAX"), flight("SNA", "LAX", month=2)])
    with pytest.raises(ValueError, match="exactly one month"):
        ingest.read_ontime(path)


def test_not_a_zip_is_rejected(tmp_path):
    path = tmp_path / "ontime_2025_01.zip"
    path.write_text("<html>404</html>", encoding="utf-8")
    with pytest.raises(ValueError, match="not a valid ZIP"):
        ingest.read_ontime(path)


def test_row_check_reports_differences(conn, ontime_dir):
    ingest.ingest_ontime(conn, ontime_dir)
    assert ingest.ontime_row_check(conn, {"2025": 5, "2025-01": 6}) == [
        "ontime 2025: 5 flights, expected 5: OK",
        "ontime 2025-01: 5 flights, expected 6: DIFFERENCE -1",
    ]


def test_row_check_skips_periods_without_a_reference(conn, ontime_dir):
    """Expectations are data: a month with no reference count (a 2026 update) is not checked."""
    ingest.ingest_ontime(conn, ontime_dir)
    assert ingest.ontime_row_check(conn, {"2025-01": 5}, periods=["2026-01"]) == []
    assert ingest.ontime_row_check(conn, {"2025-01": 5}, periods=["2025-01"]) == [
        "ontime 2025-01: 5 flights, expected 5: OK"]


def test_unmatched_ontime_airports(conn, ontime_dir):
    ingest.ingest_ontime(conn, ontime_dir)
    conn.execute("INSERT INTO airports (iata, name) VALUES ('LAX', 'Los Angeles')")
    assert store.unmatched_ontime_airports(conn) == ["SNA"]
