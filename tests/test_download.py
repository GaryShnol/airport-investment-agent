"""BTS downloads for `ingest --update` with a stubbed HTTP session (no network)."""
import io
import zipfile

import pytest
import requests

from src import download, ingest
from src.config import Settings, load_sources
from tests.conftest import T100_HEADER, t100_row
from tests.test_ingest_ontime import FLIGHTS, HEADER, write_zip

SOURCES = load_sources(Settings.from_env({}).sources_path)
PREZIP_OF = {ours: src for src, ours in SOURCES.ontime_prezip_columns.items()}
PAGE = ('<form method="post" action="./DL_SelectFields.aspx" id="form1">'
        '<input type="hidden" name="__VIEWSTATE" id="__VIEWSTATE" value="vs123" />'
        '<input type="hidden" name="__VIEWSTATEGENERATOR" id="__VIEWSTATEGENERATOR" value="gen9" />'
        '<input type="hidden" name="__EVENTVALIDATION" id="__EVENTVALIDATION" value="ev456" />'
        '<span>Latest Available Data: July 2026</span></form>')


def prezip_bytes(rows=FLIGHTS, drop=(), extra_csv=False) -> bytes:
    """A PREZIP-style ZIP: CamelCase header (from the mapping), a trailing empty column, a readme."""
    header = [PREZIP_OF.get(c, c) for c in HEADER.split(",")]
    keep = [i for i, c in enumerate(header) if c not in drop]
    lines = [",".join([header[i] for i in keep] + [""])]
    lines += [",".join([r.split(",")[i] for i in keep] + [""]) for r in rows]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2025_1.csv", "\r\n".join(lines) + "\r\n")
        z.writestr("readme.html", "<html>readme</html>")
        if extra_csv:
            z.writestr("other.csv", "a,b\n")
    return buf.getvalue()


def t100_zip_bytes(rows, header=T100_HEADER, member=SOURCES.t100_zip_csv) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("Documentation.csv", "Field,Description\n")
        z.writestr(member, header + "".join(rows))
    return buf.getvalue()


T100_ROWS = [t100_row("10.00", "1000.00", "800.00", "261.00", "ANC", "FAI", "1", 2026, 1),
             t100_row("0.00", "0.00", "0.00", "0.00", "ANC", "ANC", "1", 2026, 1)]


class FakeResponse:
    def __init__(self, status=200, content=b"", headers=None):
        self.status_code, self.content, self.headers = status, content, headers or {}
        self.text = content.decode("utf-8", "replace")

    def iter_content(self, size):
        for i in range(0, len(self.content), size):
            yield self.content[i:i + size]


class FakeSession:
    """Answers each request with the next scripted item (a response or an exception); records calls."""
    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    def request(self, method, url, data=None, timeout=None, stream=False, allow_redirects=True):
        self.calls.append((method, url, data, timeout))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def client(session, retries=2, max_bytes=10_000_000):
    pauses = []
    c = download.PoliteClient(session, timeout_s=7, delay_s=1.5, retries=retries, max_bytes=max_bytes,
                              sleep=pauses.append)
    return c, pauses


ZIP = {"Content-Type": "application/x-zip-compressed"}
URL = Settings.from_env({}).ontime_prezip_url
REQUIRED = list(SOURCES.ontime_prezip_columns)


def test_prezip_columns_renamed_give_the_same_aggregate_as_transtats(tmp_path):
    """The PREZIP mapping from the probe: same flights, CamelCase names -> identical airport table."""
    transtats = ingest.read_ontime(write_zip(tmp_path / "a" / "ontime_2025_01.zip", FLIGHTS))
    path = tmp_path / "ontime_2025_01.zip"
    path.write_bytes(prezip_bytes())
    prezip = ingest.read_ontime(path, rename=SOURCES.ontime_prezip_columns)
    assert prezip.table.equals(transtats.table)
    assert (prezip.period, prezip.flights, prezip.cancelled, prezip.diverted) == ("2025-01", 5, 1, 1)


def test_prezip_without_the_mapping_is_rejected_by_column(tmp_path):
    path = tmp_path / "ontime_2025_01.zip"
    path.write_bytes(prezip_bytes())
    with pytest.raises(ValueError, match="lacks needed columns: YEAR, MONTH"):
        ingest.read_ontime(path)


def test_fetch_ontime_ok(tmp_path):
    session = FakeSession(FakeResponse(200, prezip_bytes(), ZIP))
    c, pauses = client(session)
    path = download.fetch_ontime(c, URL, "2026-01", tmp_path, REQUIRED)
    assert path.name == "ontime_2026_01.zip" and zipfile.is_zipfile(path)
    method, url, _, timeout = session.calls[0]
    assert (method, timeout, pauses) == ("GET", 7, []) and url.endswith("_2026_1.zip")


def test_fetch_ontime_rejects_an_html_page(tmp_path):
    page = b"<!DOCTYPE html><html><body>Service unavailable</body></html>"
    c, _ = client(FakeSession(FakeResponse(200, page, {"Content-Type": "text/html"})))
    with pytest.raises(download.DownloadError, match="an HTML page"):
        download.fetch_ontime(c, URL, "2026-01", tmp_path, REQUIRED)


def test_fetch_ontime_rejects_a_missing_column(tmp_path):
    c, _ = client(FakeSession(FakeResponse(200, prezip_bytes(drop=("NASDelay",)), ZIP)))
    with pytest.raises(download.DownloadError, match="lacks needed columns: NASDelay"):
        download.fetch_ontime(c, URL, "2026-01", tmp_path, REQUIRED)


def test_fetch_ontime_rejects_two_csvs(tmp_path):
    c, _ = client(FakeSession(FakeResponse(200, prezip_bytes(extra_csv=True), ZIP)))
    with pytest.raises(download.DownloadError, match="exactly one CSV"):
        download.fetch_ontime(c, URL, "2026-01", tmp_path, REQUIRED)


def test_404_is_not_published_and_not_retried(tmp_path):
    session = FakeSession(FakeResponse(404, b"<html>404</html>"))
    c, _ = client(session)
    with pytest.raises(download.DownloadError, match="not published"):
        download.fetch_ontime(c, URL, "2026-08", tmp_path, REQUIRED)
    assert len(session.calls) == 1


def test_network_errors_and_5xx_are_retried_with_growing_pauses(tmp_path):
    session = FakeSession(requests.ConnectionError("reset"), FakeResponse(503), FakeResponse(200, prezip_bytes(), ZIP))
    c, pauses = client(session, retries=2)
    download.fetch_ontime(c, URL, "2026-01", tmp_path, REQUIRED)
    assert len(session.calls) == 3 and pauses == [3.0, 4.5]


def test_retries_are_bounded_and_leave_no_partial_file(tmp_path):
    session = FakeSession(*[requests.Timeout("slow")] * 3)
    c, _ = client(session, retries=2)
    with pytest.raises(download.DownloadError, match="failed after 3 attempts \\(Timeout\\)"):
        download.fetch_ontime(c, URL, "2026-01", tmp_path, REQUIRED)
    assert len(session.calls) == 3 and list(tmp_path.iterdir()) == []


def test_size_cap(tmp_path):
    big = FakeResponse(200, b"x" * 50, {"Content-Length": "50"})
    c, _ = client(FakeSession(big), max_bytes=10)
    with pytest.raises(download.DownloadError, match="cap"):
        download.fetch_ontime(c, URL, "2026-01", tmp_path, REQUIRED)
    unannounced = FakeResponse(200, b"x" * 50)
    c, _ = client(FakeSession(unannounced), max_bytes=10)
    with pytest.raises(download.DownloadError, match="cap"):
        download.fetch_ontime(c, URL, "2026-01", tmp_path, REQUIRED)
    assert list(tmp_path.iterdir()) == []


def test_form_state_parsed_and_missing_field_named():
    assert download.form_state(PAGE) == {"__VIEWSTATE": "vs123", "__VIEWSTATEGENERATOR": "gen9",
                                         "__EVENTVALIDATION": "ev456"}
    with pytest.raises(download.DownloadError, match="__EVENTVALIDATION"):
        download.form_state(PAGE.replace("__EVENTVALIDATION", "x"))


def fetch_t100(session, tmp_path, required=ingest.T100_COLUMNS):
    c, pauses = client(session)
    page_url = Settings.from_env({}).t100_page_url
    path = download.fetch_t100(c, page_url, SOURCES.t100_form, SOURCES.t100_fields, "2026-01", tmp_path,
                               SOURCES.t100_zip_csv, required)
    return path, pauses


def test_t100_form_flow(tmp_path):
    """GET the page for its state fields, then POST the month with the configured form values."""
    session = FakeSession(FakeResponse(200, PAGE.encode()),
                          FakeResponse(200, t100_zip_bytes(T100_ROWS), {"Content-Type": "application/zip"}))
    path, pauses = fetch_t100(session, tmp_path)
    (m1, u1, d1, _), (m2, u2, data, _) = session.calls
    assert (m1, d1, m2, u1 == u2) == ("GET", None, "POST", True)
    assert {k: data[k] for k in ("__VIEWSTATE", "__VIEWSTATEGENERATOR", "__EVENTVALIDATION")} == {
        "__VIEWSTATE": "vs123", "__VIEWSTATEGENERATOR": "gen9", "__EVENTVALIDATION": "ev456"}
    assert (data["cboYear"], data["cboPeriod"], data["cboGeography"], data["chkDownloadZip"],
            data["btnDownload"]) == ("2026", "1", "All", "on", "Download")
    assert all(data[f] == "on" for f in SOURCES.t100_fields) and len(SOURCES.t100_fields) == 19
    assert pauses == [1.5]
    assert path.name == "t100_segment_2026_01.csv" and path.match(ingest.T100_GLOB)
    assert [p.name for p in tmp_path.iterdir()] == [path.name]   # the ZIP is removed after extraction
    df = ingest.read_t100(path)
    assert len(df) == 1 and int(df["passengers"].sum()) == 800


def test_t100_html_answer_and_missing_member_and_column_rejected(tmp_path):
    session = FakeSession(FakeResponse(200, PAGE.encode()), FakeResponse(200, b"<html>error</html>"))
    with pytest.raises(download.DownloadError, match="HTML page"):
        fetch_t100(session, tmp_path)
    session = FakeSession(FakeResponse(200, PAGE.encode()), FakeResponse(200, t100_zip_bytes(T100_ROWS, member="x.csv")))
    with pytest.raises(download.DownloadError, match="has no T_T100_SEGMENT_ALL_CARRIER.csv"):
        fetch_t100(session, tmp_path)
    no_pax = T100_HEADER.replace("PASSENGERS,", "PAX,")
    session = FakeSession(FakeResponse(200, PAGE.encode()), FakeResponse(200, t100_zip_bytes(T100_ROWS, header=no_pax)))
    with pytest.raises(download.DownloadError, match="lacks needed columns: PASSENGERS"):
        fetch_t100(session, tmp_path)


def test_loaders_refuse_another_month_before_writing(conn, tmp_path):
    path = tmp_path / "ontime_2025_01.zip"
    path.write_bytes(prezip_bytes())
    with pytest.raises(ValueError, match="expected 2026-01, found 2025-01"):
        ingest.load_ontime_file(conn, path, rename=SOURCES.ontime_prezip_columns, expect_period="2026-01")
    csv = tmp_path / "t100_segment_2026_02.csv"
    csv.write_text(T100_HEADER + "".join(T100_ROWS), encoding="utf-8")
    with pytest.raises(ValueError, match="expected only 2026-02"):
        ingest.load_t100_file(conn, csv, expect_period="2026-02")
    assert conn.execute("SELECT COUNT(*) FROM ingest_log").fetchone()[0] == 0
