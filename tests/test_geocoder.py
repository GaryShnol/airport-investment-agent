"""Airport -> CBSA via the Census Geocoder, with a stubbed HTTP layer (no network)."""
from dataclasses import replace

import pytest
import requests

from src import census, ingest, store
from src.config import Settings
from tests.conftest import seed_analysis_db
from tests.test_ingest_census import FakeResponse


def geographies(metro=None, micro=None):
    g = {}
    if metro:
        g["Metropolitan Statistical Areas"] = [{"CBSA": metro[0], "GEOID": metro[0], "NAME": metro[1]}]
    if micro:
        g["Micropolitan Statistical Areas"] = [{"CBSA": micro[0], "GEOID": micro[0], "NAME": micro[1]}]
    return FakeResponse(payload={"result": {"geographies": g}})


# Peer set with min 1,000 passengers: ANC, BOS, BTV, LAX, MDW, ORD, PWM, SNA (HYA is below the floor).
COORDS = {"BOS": (42.364, -71.005), "PWM": (43.646, -70.309), "BTV": (44.472, -73.153), "HYA": (41.669, -70.280),
          "ANC": (61.174, -149.996), "LAX": (33.942, -118.408), "SNA": (33.676, -117.868),
          "ORD": (41.979, -87.905), "MDW": (41.786, -87.752)}
ANSWERS = {"BOS": geographies(metro=("14460", "Boston-Cambridge-Newton, MA-NH Metro Area")),
           "PWM": geographies(metro=("38860", "Portland-South Portland, ME Metro Area")),
           "BTV": geographies(micro=("99999", "Example Micro Area")),
           "ANC": geographies()}       # outside every CBSA in this stub


class GeoSession:
    """Answers by the coordinates sent; unknown points get a metro of their own."""
    def __init__(self, fail=(), coords=None):
        self.calls, self.fail, self.coords = [], set(fail), coords or COORDS

    def get(self, url, params=None, timeout=None, **_):
        iata = next(code for code, point in self.coords.items() if (params["y"], params["x"]) == point)
        self.calls.append(iata)
        if iata in self.fail:
            raise requests.Timeout("timed out")
        return ANSWERS.get(iata) or geographies(metro=(f"1{len(iata)}{iata}", f"{iata} Metro Area"))


@pytest.fixture
def db(conn):
    seed_analysis_db(conn)
    conn.executemany("UPDATE airports SET lat = ?, lon = ? WHERE iata = ?",
                     [(lat, lon, iata) for iata, (lat, lon) in COORDS.items()])
    return conn


@pytest.fixture
def settings():
    return Settings.from_env({})


@pytest.fixture
def run(db, settings, scoring_cfg):
    sleeps = []

    def go(session):
        return ingest.geocode_peer_airports(db, session, settings, scoring_cfg, sleep=sleeps.append)
    go.sleeps = sleeps
    return go


def test_parse_geocoder_metro_micro_and_none():
    assert census.parse_geocoder(ANSWERS["BOS"].json()) == census.GeocodeResult(
        "14460", "Boston-Cambridge-Newton, MA-NH Metro Area", "metro")
    assert census.parse_geocoder(ANSWERS["BTV"].json()).kind == "micro"
    assert census.parse_geocoder(ANSWERS["ANC"].json()) == census.GeocodeResult(None, None, None)
    with pytest.raises(KeyError):
        census.parse_geocoder({"errors": ["bad"]})


def test_peer_airports_geocoded_and_reported(db, run):
    summary = run(GeoSession())
    assert summary.no_cbsa == ["ANC"]
    assert summary.matched == ["BOS", "BTV", "LAX", "MDW", "ORD", "PWM", "SNA"]
    assert "HYA" not in summary.matched + summary.no_cbsa  # below the peer-set floor: never requested
    assert summary.requested == 8 and summary.failed == []
    assert len(run.sleeps) == 7  # a pause between requests, none before the first
    assert db.execute("SELECT cbsa FROM airports WHERE iata = 'BOS'").fetchone() == ("14460",)
    assert db.execute("SELECT cbsa, kind FROM airport_cbsa WHERE iata = 'ANC'").fetchone() == (None, None)


def test_rerun_uses_the_cache_including_no_match(run):
    run(GeoSession())
    again = GeoSession()
    summary = run(again)
    assert again.calls == [] and summary.requested == 0
    assert summary.no_cbsa == ["ANC"] and len(summary.matched) == 7


def test_moved_coordinates_or_new_vintage_are_geocoded_again(db, run, settings, scoring_cfg):
    run(GeoSession())
    db.execute("UPDATE airports SET lat = 42.365, lon = -71.006 WHERE iata = 'BOS'")
    moved = GeoSession(coords={**COORDS, "BOS": (42.365, -71.006)})
    run(moved)
    assert moved.calls == ["BOS"]
    new_vintage = GeoSession(coords=moved.coords)
    ingest.geocode_peer_airports(db, new_vintage, settings, replace(scoring_cfg, geocoder_vintage="ACS2025_Current"),
                                 sleep=lambda s: None)
    assert len(new_vintage.calls) == 8


def test_failure_is_tolerated_not_cached_and_retried(db, run):
    summary = run(GeoSession(fail={"LAX"}))
    assert summary.failed == ["LAX"] and len(summary.matched) == 6
    assert db.execute("SELECT COUNT(*) FROM airport_cbsa WHERE iata = 'LAX'").fetchone() == (0,)
    retry = GeoSession()
    assert run(retry).failed == [] and retry.calls == ["LAX"]


def test_stops_after_consecutive_failures(run, monkeypatch):
    monkeypatch.setattr(ingest, "GEOCODER_MAX_CONSECUTIVE_FAILURES", 2)
    session = GeoSession(fail=set(COORDS))
    summary = run(session)
    assert len(session.calls) == 2 and len(summary.failed) == 8


def test_airports_reload_keeps_cbsa_through_the_cache(db, run, scoring_cfg):
    run(GeoSession())
    db.execute("UPDATE airports SET cbsa = NULL")  # what ingest_airports does on every run
    assert store.apply_cbsa(db, scoring_cfg.geocoder_vintage) == 7
