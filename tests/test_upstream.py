"""ingest --check and freshness states, with a stubbed HTTP layer (no network)."""
from datetime import date

import pytest
import requests

from src import ingest, store, upstream
from src.config import ROOT, Settings, load_scoring
from tests.test_ingest_census import FakeResponse

TODAY = date(2026, 10, 10)
ZIP = {"Content-Type": "application/x-zip-compressed"}
HTML = {"Content-Type": "text/html"}
JSON = {"Content-Type": "application/json;charset=utf-8"}
T100_PAGE = '<td>Latest Available Data: </td><td><b>July 2026</b></td>'


class Upstream:
    """HEAD for On-Time months (published up to `ontime_latest`), GET for the T-100 page and Census
    metadata (published up to `census_latest`). `down` makes every request fail."""
    def __init__(self, ontime_latest="2026-07", census_latest=2024, t100_page=T100_PAGE, down=False):
        self.ontime_latest, self.census_latest, self.t100_page, self.down = ontime_latest, census_latest, t100_page, down
        self.heads, self.gets = [], []

    def head(self, url, timeout=None, allow_redirects=True):
        self.heads.append(url)
        if self.down:
            raise requests.ConnectionError("down")
        year, month = url.rsplit("_", 2)[-2:]
        period = f"{int(year):04d}-{int(month.removesuffix('.zip')):02d}"
        return FakeResponse(200, headers=ZIP) if period <= self.ontime_latest else FakeResponse(404, headers=HTML)

    def get(self, url, params=None, timeout=None):
        self.gets.append(url)
        if self.down:
            raise requests.Timeout("down")
        if "DL_SelectFields" in url:
            return FakeResponse(200, self.t100_page, headers=HTML)
        year = int(url.split("/data/")[1].split("/")[0])
        return FakeResponse(200, "{}", headers=JSON) if year <= self.census_latest else FakeResponse(404, headers=HTML)


@pytest.fixture
def settings():
    return Settings.from_env({})


@pytest.fixture
def cfg():
    return load_scoring(ROOT / "config" / "weights.yaml")


def test_ontime_walks_months_until_404(settings):
    s = Upstream()
    assert upstream.check_ontime(s, settings.ontime_prezip_url, "2025-12", TODAY, 30) == "2026-07"
    assert len(s.heads) == 8 and s.heads[-1].endswith("_2026_8.zip")  # 2026-01..2026-07 published, 2026-08 404


def test_ontime_nothing_newer_is_the_loaded_month(settings):
    assert upstream.check_ontime(Upstream(ontime_latest="2025-12"), settings.ontime_prezip_url, "2025-12",
                                 TODAY, 30) == "2025-12"


@pytest.mark.parametrize("response", [FakeResponse(200, "<html>", headers=HTML), FakeResponse(503, headers=HTML)])
def test_ontime_unexpected_answer_is_unknown(settings, response):
    class Odd(Upstream):
        def head(self, url, timeout=None, allow_redirects=True):
            return response
    assert upstream.check_ontime(Odd(), settings.ontime_prezip_url, "2025-12", TODAY, 30) is None


@pytest.mark.parametrize("html, expected", [
    (T100_PAGE, "2026-07"), ("Latest Available Data: December 2025", "2025-12"),
    ("<html>maintenance</html>", None), ("Latest Available Data: Smarch 2026", None)])
def test_t100_page_label(html, expected):
    assert upstream.parse_t100_latest(html) == expected


def test_census_newest_vintage(settings, cfg):
    s = Upstream(census_latest=2025)
    assert upstream.check_census(s, settings.census_api_url, cfg.census_dataset, cfg.census_variable, 2024,
                                 TODAY, 30) == "2025"
    assert s.gets[-1].endswith("/2026/acs/acs5/variables/B01003_001E.json")


def loaded_db(conn):
    conn.executemany("INSERT INTO meta (source, last_period) VALUES (?, ?)",
                     [("t100", "2025-12"), ("ontime", "2025-12"), ("census", "2024")])
    return conn


def test_run_check_stores_result_with_timestamp(conn, settings, cfg):
    rows = upstream.run_check(loaded_db(conn), Upstream(), settings, cfg, TODAY)
    assert {(r.source, r.upstream, r.freshness) for r in rows} == {
        ("ontime", "2026-07", "stale (upstream has 2026-07)"),
        ("t100", "2026-07", "stale (upstream has 2026-07)"),
        ("census", "2024", "fresh")}
    meta = {r["source"]: r for r in store.meta_rows(conn)}
    assert meta["ontime"]["upstream_latest"] == "2026-07" and meta["ontime"]["checked_at"]
    assert store.overall_freshness(store.meta_rows(conn)) == \
        "stale (upstream has 2024 for census, 2026-07 for ontime, 2026-07 for t100)".replace("2024 for census, ", "")


def test_all_fresh(conn, settings, cfg):
    upstream.run_check(loaded_db(conn), Upstream(ontime_latest="2025-12", t100_page="Latest Available Data: "
                                                 "December 2025"), settings, cfg, TODAY)
    assert store.overall_freshness(store.meta_rows(conn)) == "fresh"


def test_network_failure_gives_unknown_not_an_error(conn, settings, cfg):
    rows = upstream.run_check(loaded_db(conn), Upstream(down=True), settings, cfg, TODAY)
    assert {r.freshness for r in rows} == {"unknown"}
    assert all(r["checked_at"] and r["upstream_latest"] is None for r in store.meta_rows(conn))
    assert store.overall_freshness(store.meta_rows(conn)) == "unknown"


def test_never_checked_is_unknown():
    assert store.source_freshness({"source": "t100", "last_period": "2025-12", "upstream_latest": None,
                                   "checked_at": None}) == "unknown"
    assert store.overall_freshness([]) == "unknown"


def test_check_cli_prints_a_table(tmp_path, capsys, monkeypatch):
    db = tmp_path / "a.db"
    with store.connect_rw(db) as c:
        store.init_db(c)
        loaded_db(c)
    settings = Settings.from_env({"DB_PATH": str(db)})
    monkeypatch.setattr(ingest.census, "session", lambda: Upstream())
    ingest.check(settings)
    out = capsys.readouterr().out
    assert "ontime   2025-12   2026-07   stale (upstream has 2026-07)" in out
    assert out.strip().endswith("overall: stale (upstream has 2026-07 for ontime, 2026-07 for t100)")


def test_chat_never_checks_upstream(monkeypatch):
    """Tools read freshness from meta only: no tool module imports the network check."""
    import src.tools
    assert "upstream" not in vars(src.tools)
    assert not hasattr(src.tools, "requests")
