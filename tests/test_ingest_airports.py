import pytest

from src import ingest
from src.config import Settings

AIRPORTS_HEADER = (
    '"id","ident","type","name","latitude_deg","longitude_deg","elevation_ft","continent","iso_country",'
    '"iso_region","municipality","scheduled_service","icao_code","iata_code","gps_code","local_code",'
    '"home_link","wikipedia_link","keywords"\n'
)


def airport_row(ident, kind, name, country, region, iata, scheduled="yes", lat=42.36, lon=-71.0, keywords=""):
    return (
        f'1,"{ident}","{kind}","{name}",{lat},{lon},0,"NA","{country}","{region}","x","{scheduled}",,"{iata}",,,,,'
        f'"{keywords}"\n'
    )


@pytest.fixture
def airports_csv(tmp_path):
    path = tmp_path / "airports.csv"
    path.write_text(
        AIRPORTS_HEADER
        + airport_row("KBOS", "large_airport", "Logan", "US", "US-MA", "BOS")
        + airport_row("PANC", "large_airport", "Ted Stevens", "US", "US-AK", "ANC", lat=61.17, lon=-150.0)
        + airport_row("00A", "heliport", "No IATA", "US", "US-PA", "")
        + airport_row("TJSJ", "large_airport", "Luis Munoz Marin", "PR", "PR-U-A", "SJU")
        + airport_row("CYVR", "large_airport", "Vancouver", "CA", "CA-BC", "YVR")
        # duplicate IATA: the closed one must lose to the scheduled one
        + airport_row("XOLD", "closed", "Old Field", "US", "US-TX", "DUP", scheduled="no")
        + airport_row("XNEW", "medium_airport", "New Field", "US", "US-TX", "DUP"),
        encoding="utf-8",
    )
    return path


def airports(conn):
    return {r[0]: r[1:] for r in conn.execute("SELECT iata, name, state, region, cbsa, lat, lon FROM airports")}


def test_airports_without_iata_skipped(conn, airports_csv):
    n = ingest.ingest_airports(conn, airports_csv, {"new_england": ("MA", "CT")})
    assert n == 5
    assert "" not in airports(conn) and None not in airports(conn)


def test_airports_state_region_and_coords(conn, airports_csv):
    ingest.ingest_airports(conn, airports_csv, {"new_england": ("MA", "CT")})
    a = airports(conn)
    assert a["BOS"] == ("Logan", "MA", "new_england", None, 42.36, -71.0)
    assert a["ANC"][1:3] == ("AK", None)
    assert a["SJU"][1] == "PR"  # US territory: state = country code, as in T-100
    assert a["YVR"][1] is None  # foreign: no state


def test_airports_duplicate_iata_prefers_scheduled(conn, airports_csv):
    ingest.ingest_airports(conn, airports_csv, {})
    assert airports(conn)["DUP"][0] == "New Field"


def test_airports_rerun_is_idempotent(conn, airports_csv):
    ingest.ingest_airports(conn, airports_csv, {})
    assert ingest.ingest_airports(conn, airports_csv, {}) == 5
    assert conn.execute("SELECT COUNT(*) FROM airports").fetchone() == (5,)


def test_main_runs_end_to_end(tmp_path, t100_dir, airports_csv, monkeypatch, capsys):
    raw = t100_dir.parent
    (raw / "airports.csv").write_text(airports_csv.read_text(encoding="utf-8"), encoding="utf-8")


    settings = Settings.from_env({"DB_PATH": str(tmp_path / "out.db"), "RAW_DATA_DIR": str(raw)})
    monkeypatch.setattr(ingest, "get_settings", lambda: settings)
    ingest.main([])
    ingest.main([])
    out = capsys.readouterr().out
    assert "t100: 5 rows loaded" in out and "t100: 0 rows loaded" in out
    assert "origins without an airports match: 0" in out  # ANC and YVR are both in airports.csv


def test_airports_old_code_from_t100_resolved_by_keyword(conn, tmp_path):
    # OurAirports recoded PBI to DJT; BTS still reports PBI. The old code survives in keywords.
    conn.execute("INSERT INTO t100_route_month VALUES ('PBI','FL','ATL',2025,1,'pax',545,1,100,90)")
    path = tmp_path / "a.csv"
    path.write_text(
        AIRPORTS_HEADER
        + airport_row("KPBI", "large_airport", "Palm Beach", "US", "US-FL", "DJT", keywords="PBI, KPBI")
        # keyword on a field without scheduled service or IATA code is ignored (too noisy)
        + airport_row("XATL", "closed", "Old Atlanta", "US", "US-GA", "", scheduled="no", keywords="ATL"),
        encoding="utf-8",
    )
    assert ingest.ingest_airports(conn, path, {}) == 2
    a = airports(conn)
    assert a["PBI"][:2] == ("Palm Beach", "FL") and a["DJT"][0] == "Palm Beach"
    assert "ATL" not in a
