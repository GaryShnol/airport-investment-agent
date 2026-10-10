"""`ingest --update` end to end with a stubbed BTS (no network): plan, confirmation, months, failures."""
from datetime import date

import pytest

from src import ingest, store, update
from src.config import ROOT, Settings, load_scoring, load_sources
from src.ingest import GeocodeSummary
from tests.conftest import t100_row
from tests.test_download import PAGE, FakeResponse, prezip_bytes, t100_zip_bytes
from tests.test_ingest_ontime import flight

TODAY = date(2026, 10, 10)
MONTHS = {1: "January", 2: "February", 3: "March"}
SETTINGS = Settings.from_env({})
CFG = load_scoring(ROOT / "config" / "weights.yaml")
SRC = load_sources(SETTINGS.sources_path)


def month_rows(period):
    year, month = map(int, period.split("-"))
    rows = [flight("SNA", "LAX", month=month), flight("LAX", "SNA", dep_del15="1.00", month=month)]
    return [",".join([str(year), *r.split(",")[1:]]) for r in rows]


class Bts:
    """HEAD for the On-Time check, GET for the T-100 page (label and form) and requests for the
    downloads. Months up to `latest` are published; `broken` maps (source, period) to a bad answer."""
    def __init__(self, latest="2026-02", broken=None):
        self.latest, self.broken = latest, broken or {}
        self.downloads = []

    def head(self, url, timeout=None, allow_redirects=True):
        year, month = url.removesuffix(".zip").rsplit("_", 2)[-2:]
        published = f"{int(year):04d}-{int(month):02d}" <= self.latest
        return FakeResponse(200 if published else 404, headers={"Content-Type": "application/x-zip-compressed"})

    def page(self):
        year, month = map(int, self.latest.split("-"))
        return PAGE.replace("July 2026", f"{MONTHS[month]} {year}")

    def get(self, url, params=None, timeout=None):
        return FakeResponse(200, self.page().encode(), {"Content-Type": "text/html"})

    def request(self, method, url, data=None, timeout=None, stream=False, allow_redirects=True):
        if method == "GET" and "DL_SelectFields" in url:
            return FakeResponse(200, self.page().encode())
        if method == "POST":
            period = f"{int(data['cboYear']):04d}-{int(data['cboPeriod']):02d}"
            source = "t100"
            year, month = map(int, period.split("-"))
            body = t100_zip_bytes([t100_row("10.00", "1000.00", "800.00", "261.00", "ANC", "FAI", "1", year, month),
                                   t100_row("2.00", "0.00", "0.00", "2846.00", "ANC", "ORD", "2", year, month)])
        else:
            year, month = url.removesuffix(".zip").rsplit("_", 2)[-2:]
            period, source = f"{int(year):04d}-{int(month):02d}", "ontime"
            body = prezip_bytes(month_rows(period))
        self.downloads.append((source, period))
        return self.broken.get((source, period)) or FakeResponse(200, body, {"Content-Type": "application/zip"})


@pytest.fixture
def db(conn):
    """Loaded through 2025-12 (log and meta only: enough for plans and the window)."""
    for year in (2024, 2025):
        for month in range(1, 13):
            conn.execute("INSERT INTO ingest_log (source, period, rows, loaded_at) VALUES ('t100', ?, 1, 'x')",
                         (f"{year}-{month:02d}",))
            if year == 2025:
                conn.execute("INSERT INTO ingest_log (source, period, rows, loaded_at) VALUES ('ontime', ?, 1, 'x')",
                             (f"{year}-{month:02d}",))
    ingest.update_meta(conn, "t100")
    ingest.update_meta(conn, "ontime")
    conn.commit()
    return conn


def run(conn, bts, **kw):
    lines, pauses = [], []
    kw.setdefault("interactive", False)
    code = update.run(conn, SETTINGS, CFG, SRC, session=bts, sleep=pauses.append, today=TODAY, out=lines.append,
                      geocode=lambda *a: GeocodeSummary(matched=["ANC"], requested=0), **kw)
    return code, "\n".join(lines), pauses


def test_plan_months():
    plan = update.plan_source
    assert update.months_between("2025-12", "2026-03") == ["2026-01", "2026-02", "2026-03"]

    class Conn:   # loaded_periods only
        def __init__(self, periods): self.periods = periods
        def execute(self, *_): return [(p,) for p in self.periods]
    c = Conn({"2025-12"})
    assert plan(c, "ontime", "2025-12", "2026-07", None).months == tuple(f"2026-{m:02d}" for m in range(1, 8))
    assert plan(c, "ontime", "2025-12", "2026-07", "2026-01").months == ("2026-01",)
    capped = plan(c, "t100", "2025-12", "2026-02", "2026-05")
    assert capped.months == ("2026-01", "2026-02") and "upstream has only 2026-02" in capped.note
    unknown = plan(c, "t100", "2025-12", None, None)
    assert unknown.months == () and "unknown" in unknown.note
    assert plan(c, "t100", "2025-12", None, "2026-01").months == ("2026-01",)
    assert plan(Conn({"2025-12", "2026-01"}), "ontime", "2025-12", "2026-02", None).months == ("2026-02",)
    assert plan(c, "ontime", "2025-12", "2025-12", None).months == ()


def test_dry_run_prints_the_plan_and_downloads_nothing(db):
    bts = Bts()
    code, text, _ = run(db, bts, dry_run=True)
    assert code == 0 and bts.downloads == []
    assert "ontime  loaded 2025-12, upstream 2026-02: 2 months: 2026-01..2026-02" in text
    assert "download: about 62 MB" in text and "Dry run: nothing downloaded." in text
    assert db.execute("SELECT last_period FROM meta WHERE source = 'ontime'").fetchone()[0] == "2025-12"


def test_not_interactive_without_yes_aborts(db):
    bts = Bts()
    code, text, _ = run(db, bts)
    assert code == 2 and bts.downloads == [] and "rerun with --yes" in text


def test_confirmation(db):
    bts = Bts()
    code, text, _ = run(db, bts, interactive=True, ask=lambda _: "n")
    assert code == 1 and bts.downloads == [] and "Aborted: nothing downloaded." in text
    code, _, _ = run(db, bts, interactive=True, ask=lambda _: "y", through="2026-01")
    assert code == 0 and bts.downloads == [("ontime", "2026-01"), ("t100", "2026-01")]


def test_update_loads_month_by_month_and_moves_the_window(db):
    bts = Bts()
    code, text, pauses = run(db, bts, yes=True)
    assert code == 0
    assert bts.downloads == [("ontime", "2026-01"), ("t100", "2026-01"), ("ontime", "2026-02"), ("t100", "2026-02")]
    assert pauses and all(p == SETTINGS.download_delay_s for p in pauses)   # a pause before every later request
    assert "ontime  2026-01: 2 flights, 2 airports" in text
    assert "t100    2026-01: 2 rows, 0.0M passenger-service passengers" in text
    assert "loaded: ontime 2026-02, t100 2026-02" in text
    assert "analysis window: 2025-03..2026-02 (12 of 12 months), growth vs 2024-03..2025-02" in text
    assert "freshness: fresh" in text
    assert db.execute("SELECT COUNT(*) FROM ontime_airport_month WHERE year = 2026").fetchone()[0] == 4
    pax = db.execute("SELECT SUM(passengers) FROM t100_route_month WHERE year = 2026 AND config = 'pax'").fetchone()[0]
    assert pax == 1600


def test_rerun_is_idempotent(db):
    run(db, Bts(), yes=True)
    bts = Bts()
    code, text, _ = run(db, bts, yes=True)
    assert code == 0 and bts.downloads == [] and "Nothing to update." in text
    assert db.execute("SELECT COUNT(*) FROM ingest_log WHERE period LIKE '2026%'").fetchone()[0] == 4


def test_failure_on_month_two_keeps_month_one(db):
    html = FakeResponse(200, b"<html><body>Error</body></html>", {"Content-Type": "text/html"})
    bts = Bts(broken={("ontime", "2026-02"): html})
    code, text, _ = run(db, bts, yes=True)
    assert code == 1
    assert bts.downloads == [("ontime", "2026-01"), ("t100", "2026-01"), ("ontime", "2026-02")]   # stops there
    assert "ontime  2026-02: FAILED, nothing written for this month: the download is an HTML page" in text
    assert "loaded: ontime 2026-01, t100 2026-01" in text and "analysis window: 2025-02..2026-01" in text
    assert "stale (upstream has 2026-02 for ontime, 2026-02 for t100)" in text
    assert store.loaded_periods(db, "ontime") >= {"2026-01"} and "2026-02" not in store.loaded_periods(db, "ontime")
    assert db.execute("SELECT COUNT(*) FROM ontime_airport_month WHERE year = 2026 AND month = 2").fetchone()[0] == 0
    # the next run picks up where this one stopped
    bts = Bts()
    assert run(db, bts, yes=True)[0] == 0 and bts.downloads == [("ontime", "2026-02"), ("t100", "2026-02")]


def test_failure_after_writing_rolls_the_month_back(db, monkeypatch):
    """An error after the rows are inserted (here: in the meta update) leaves no trace of the month."""
    def boom(conn, source):
        raise ValueError("disk full")
    monkeypatch.setattr(ingest, "update_meta", boom)
    code, text, _ = run(db, Bts(), yes=True, through="2026-01")
    assert code == 1 and "disk full" in text
    assert db.execute("SELECT COUNT(*) FROM ontime_airport_month").fetchone()[0] == 0
    assert "2026-01" not in store.loaded_periods(db, "ontime")


def test_cli_options():
    for argv in (["--dry-run"], ["--yes"], ["--through", "2026-01"], ["--update", "--through", "2026-13"],
                 ["--check", "--update"]):
        with pytest.raises(SystemExit) as exc:
            ingest.main(argv)
        assert exc.value.code == 2


def test_cli_update_without_a_db(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ingest, "get_settings", lambda: Settings.from_env({"DB_PATH": str(tmp_path / "none.db")}))
    with pytest.raises(SystemExit) as exc:
        ingest.main(["--update", "--dry-run"])
    assert exc.value.code == 1 and "No database" in capsys.readouterr().out
    assert not (tmp_path / "none.db").exists()
