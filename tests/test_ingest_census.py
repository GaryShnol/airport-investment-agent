"""Census population ingest with a stubbed HTTP layer (no network)."""
import json
from dataclasses import replace

import pytest
import requests

from src import census, ingest, store
from src.config import ROOT, Settings, load_scoring

KEY = "secret-census-key"
GEO = census.CBSA_GEOGRAPHY


class FakeResponse:
    def __init__(self, status=200, text="", payload=None, headers=None):
        self.status_code, self.headers = status, headers or {}
        self.text = json.dumps(payload) if payload is not None else text

    def json(self):
        return json.loads(self.text)


class FakeSession:
    """Answers GET by URL substring; records every call."""
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, params=None, timeout=None, **_):
        self.calls.append((url, params))
        for part, answer in self.routes.items():
            if part in url:
                if isinstance(answer, Exception):
                    raise answer
                return answer
        return FakeResponse(404, "<html><title>Not Found</title></html>")


def population(rows):
    return FakeResponse(payload=[["NAME", "B01003_001E", GEO], *rows])


VINTAGE_2024 = population([["Boston-Cambridge-Newton, MA-NH Metro Area", "4919179", "14460"],
                           ["Lebanon-Claremont, NH-VT Micro Area", "220000", "30150"],
                           ["No estimate", "-666666666", "99999"]])
VINTAGE_2019 = population([["Boston-Cambridge-Newton, MA-NH Metro Area", "4832346", "14460"]])


@pytest.fixture
def settings():
    return Settings.from_env({"CENSUS_API_KEY": KEY})


@pytest.fixture
def cfg():
    return replace(load_scoring(ROOT / "config" / "weights.yaml"), census_vintage=2024, census_base_vintage=2019)


def test_parse_population_strings_to_ints_and_skips_annotations():
    rows = census.parse_population(200, VINTAGE_2024.text, "B01003_001E")
    assert rows == [("14460", "Boston-Cambridge-Newton, MA-NH Metro Area", 4919179),
                    ("30150", "Lebanon-Claremont, NH-VT Micro Area", 220000)]


@pytest.mark.parametrize("status, text, message", [
    (200, "<html><head><title>Missing Key</title></head></html>", "CENSUS_API_KEY"),
    (200, "<html><head><title>Invalid Key</title></head></html>", "CENSUS_API_KEY"),
    (500, "<html><title>Server Error</title></html>", "HTTP 500"),
    (200, json.dumps([["NAME", "B01003_001E", GEO]]), "no rows"),
    (200, json.dumps([["NAME", "OTHER", GEO], ["x", "1", "1"]]), "lacks columns"),
])
def test_parse_population_errors_are_clear(status, text, message):
    with pytest.raises(census.CensusError, match=message):
        census.parse_population(status, text, "B01003_001E")


def test_ingest_loads_both_vintages_with_meta(conn, settings, cfg):
    session = FakeSession({"/2024/": VINTAGE_2024, "/2019/": VINTAGE_2019})
    assert ingest.ingest_census(conn, session, settings, cfg) == [2019, 2024]
    rows = conn.execute("SELECT cbsa, year, population, name FROM census_metro ORDER BY year, cbsa").fetchall()
    assert rows == [("14460", 2019, 4832346, "Boston-Cambridge-Newton, MA-NH Metro Area"),
                    ("14460", 2024, 4919179, "Boston-Cambridge-Newton, MA-NH Metro Area"),
                    ("30150", 2024, 220000, "Lebanon-Claremont, NH-VT Micro Area")]
    last, fetched = conn.execute("SELECT last_period, fetched_at FROM meta WHERE source = 'census'").fetchone()
    assert last == "2024" and fetched
    # the key travels in the query string, the dataset and variable come from config
    url, params = session.calls[0]
    assert url.endswith("/acs/acs5") and params["key"] == KEY and params["get"] == "NAME,B01003_001E"


def test_rerun_is_idempotent_and_makes_no_request(conn, settings, cfg):
    ingest.ingest_census(conn, FakeSession({"/2024/": VINTAGE_2024, "/2019/": VINTAGE_2019}), settings, cfg)
    before = conn.execute("SELECT * FROM census_metro ORDER BY year, cbsa").fetchall()
    again = FakeSession({})
    assert ingest.ingest_census(conn, again, settings, cfg) == []
    assert again.calls == []
    assert conn.execute("SELECT * FROM census_metro ORDER BY year, cbsa").fetchall() == before


def test_failed_vintage_writes_nothing(conn, settings, cfg):
    session = FakeSession({"/2024/": VINTAGE_2024, "/2019/": FakeResponse(200, "<title>Invalid Key</title>")})
    with pytest.raises(census.CensusError, match="vintage 2019"):
        ingest.ingest_census(conn, session, settings, cfg)
    assert conn.execute("SELECT COUNT(*) FROM census_metro").fetchone() == (0,)
    assert store.loaded_periods(conn, ingest.CENSUS_SOURCE) == set()
    assert conn.execute("SELECT COUNT(*) FROM meta WHERE source = 'census'").fetchone() == (0,)


def test_network_error_never_shows_the_key(conn, settings, cfg):
    error = requests.ConnectionError(f"Max retries exceeded with url: /data/2024/acs/acs5?key={KEY}")
    with pytest.raises(census.CensusError) as exc:
        ingest.ingest_census(conn, FakeSession({"/2024/": error}), settings, cfg)
    assert KEY not in str(exc.value) and "ConnectionError" in str(exc.value)


def test_census_never_sets_the_analysis_window(conn, settings, cfg):
    """A yearly vintage ('2024') would sort before '2025-12' and shift the window if it counted."""
    conn.execute("INSERT INTO meta (source, last_period) VALUES ('t100', '2025-12')")
    ingest.ingest_census(conn, FakeSession({"/2024/": VINTAGE_2024, "/2019/": VINTAGE_2019}), settings, cfg)
    assert store.analysis_window(conn).end == "2025-12"
