"""Bootstrap (`src.bootstrap`) with a stubbed HTTP layer (no network)."""
import gzip
import hashlib

import pytest
import requests

from src import bootstrap, store
from src.config import Settings
from tests.conftest import full_db

URL = "https://example.test/releases/download/data-2025/airport.db.gz?token=s3cret"


class FakeResponse:
    def __init__(self, status=200, body=b"", headers=None, cut_after=None):
        self.status_code, self.body, self.headers, self.cut_after = status, body, headers or {}, cut_after

    @property
    def text(self):
        return self.body.decode()

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def iter_content(self, size):
        size = min(size, 1024)   # small chunks, so a cut can fall in the middle of the file
        for i in range(0, len(self.body), size):
            if self.cut_after is not None and i >= self.cut_after:
                raise requests.exceptions.ChunkedEncodingError("connection broken")
            yield self.body[i:i + size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    """Serves `files` by URL (without the query string); records every URL requested."""
    def __init__(self, files=None, cut_after=None, lie_length=False):
        self.files, self.cut_after, self.lie_length, self.urls = files or {}, cut_after, lie_length, []

    def get(self, url, stream=False, timeout=None):
        self.urls.append(url)
        body = self.files.get(url.split("?")[0])
        if body is None:
            return FakeResponse(404)
        length = len(body) + 100 if self.lie_length else len(body)
        return FakeResponse(200, body, {"Content-Length": str(length)}, self.cut_after)


@pytest.fixture
def seed_gz(tmp_path):
    """A valid seed DB, gzipped, and its sha256."""
    db = full_db(tmp_path / "source.db")
    packed = gzip.compress(db.read_bytes())
    return packed, hashlib.sha256(packed).hexdigest()


def settings_for(tmp_path, **env):
    return Settings.from_env({"DB_PATH": str(tmp_path / "data" / "airport.db"), **env})


def serve(packed, sha):
    base = URL.split("?")[0]
    return FakeSession({base: packed, base + ".sha256": f"{sha}  airport.db.gz\n".encode()})


@pytest.fixture(autouse=True)
def no_freshness_check(monkeypatch):
    """The freshness check is exercised in its own tests; keep it off the network here."""
    calls = []
    monkeypatch.setattr(bootstrap.ingest, "check", lambda settings: calls.append(settings))
    return calls


def leftovers(settings):
    return [p.name for p in settings.db_path.parent.iterdir() if p.name.startswith(".bootstrap-")]


def test_valid_db_means_no_download(tmp_path):
    s = settings_for(tmp_path, SEED_DB_URL=URL)
    full_db(s.db_path)
    session = FakeSession()
    bootstrap.run(s, session)
    assert session.urls == []


def test_download_verify_and_install(tmp_path, seed_gz, capsys):
    s = settings_for(tmp_path, SEED_DB_URL=URL)
    session = serve(*seed_gz)
    bootstrap.run(s, session)
    assert store.db_problem(s.db_path) is None
    assert [u.split("?")[0] for u in session.urls] == [URL.split("?")[0] + ".sha256", URL.split("?")[0]]
    assert leftovers(s) == []
    assert "s3cret" not in capsys.readouterr().out    # query string (a token) never printed


def test_checksum_from_settings_skips_the_sha256_file(tmp_path, seed_gz):
    packed, sha = seed_gz
    s = settings_for(tmp_path, SEED_DB_URL=URL, SEED_DB_SHA256=sha.upper())
    session = FakeSession({URL.split("?")[0]: packed})
    bootstrap.run(s, session)
    assert store.db_problem(s.db_path) is None and len(session.urls) == 1


def test_bad_checksum_is_rejected_and_nothing_installed(tmp_path, seed_gz):
    packed, _ = seed_gz
    s = settings_for(tmp_path, SEED_DB_URL=URL)
    with pytest.raises(bootstrap.BootstrapError, match="checksum mismatch") as exc:
        bootstrap.run(s, serve(packed, "0" * 64))
    assert not s.db_path.exists() and leftovers(s) == []
    assert "SEED_DB_URL" in str(exc.value) and "src.ingest" in str(exc.value)   # the three options


def test_no_url_lists_the_three_options(tmp_path):
    s = settings_for(tmp_path)
    with pytest.raises(bootstrap.BootstrapError) as exc:
        bootstrap.run(s, FakeSession())
    message = str(exc.value)
    assert "SEED_DB_URL is not set" in message
    assert "1. Set SEED_DB_URL" in message and "2. Mount an existing" in message and "3. Put the raw files" in message


def test_interrupted_download_leaves_no_db(tmp_path, seed_gz):
    s = settings_for(tmp_path, SEED_DB_URL=URL)
    session = serve(*seed_gz)
    session.cut_after = 2048     # the connection breaks after 2 KB
    with pytest.raises(bootstrap.BootstrapError, match="ChunkedEncodingError"):
        bootstrap.fetch_seed(session, s)
    assert not s.db_path.exists() and leftovers(s) == []


def test_short_download_is_detected(tmp_path, seed_gz):
    s = settings_for(tmp_path, SEED_DB_URL=URL)
    session = serve(*seed_gz)
    session.lie_length = True    # server announced more bytes than it sent
    with pytest.raises(bootstrap.BootstrapError, match="was cut"):
        bootstrap.fetch_seed(session, s)
    assert not s.db_path.exists()


def test_atomic_replace_keeps_the_invalid_db_aside(tmp_path, seed_gz):
    s = settings_for(tmp_path, SEED_DB_URL=URL)
    s.db_path.parent.mkdir(parents=True)
    s.db_path.write_bytes(b"broken")
    bootstrap.run(s, serve(*seed_gz))
    assert store.db_problem(s.db_path) is None
    assert s.db_path.with_name("airport.db.invalid").read_bytes() == b"broken"
    assert leftovers(s) == []


def test_failed_install_keeps_the_existing_db_untouched(tmp_path, seed_gz):
    packed, _ = seed_gz
    s = settings_for(tmp_path, SEED_DB_URL=URL)
    s.db_path.parent.mkdir(parents=True)
    s.db_path.write_bytes(b"old")
    with pytest.raises(bootstrap.BootstrapError):
        bootstrap.run(s, serve(packed, "f" * 64))
    assert s.db_path.read_bytes() == b"old" and not s.db_path.with_name("airport.db.invalid").exists()


def test_not_gzip_or_not_a_db_is_rejected(tmp_path):
    s = settings_for(tmp_path, SEED_DB_URL=URL)
    for body, match in ((b"<html>not found</html>", "not a valid gzip"), (gzip.compress(b"x" * 5000), "not usable")):
        with pytest.raises(bootstrap.BootstrapError, match=match):
            bootstrap.fetch_seed(serve(body, hashlib.sha256(body).hexdigest()), s)
        assert not s.db_path.exists() and leftovers(s) == []


def test_missing_checksum_file(tmp_path, seed_gz):
    packed, _ = seed_gz
    s = settings_for(tmp_path, SEED_DB_URL=URL)
    with pytest.raises(bootstrap.BootstrapError, match="SEED_DB_SHA256"):
        bootstrap.fetch_seed(FakeSession({URL.split("?")[0]: packed}), s)


def test_freshness_check_runs_with_the_short_timeout(tmp_path, no_freshness_check):
    s = settings_for(tmp_path, BOOTSTRAP_CHECK_TIMEOUT_S="2")
    full_db(s.db_path)
    bootstrap.run(s, FakeSession())
    assert [c.http_timeout_s for c in no_freshness_check] == [2.0]


def test_freshness_check_failure_never_fails_startup(tmp_path, monkeypatch, capsys):
    s = settings_for(tmp_path)
    full_db(s.db_path)

    def boom(settings):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(bootstrap.ingest, "check", boom)
    bootstrap.run(s, FakeSession())
    assert "freshness unknown" in capsys.readouterr().out


def test_main_exits_cleanly_without_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bootstrap, "get_settings", lambda: settings_for(tmp_path))
    monkeypatch.setattr(bootstrap.census, "session", FakeSession)
    with pytest.raises(SystemExit) as exc:
        bootstrap.main()
    err = capsys.readouterr().err
    assert exc.value.code == 1 and "Traceback" not in err and "SEED_DB_URL is not set" in err


def test_bootstrap_settings_validation():
    from src.config import ConfigError
    with pytest.raises(ConfigError, match="SEED_DB_SHA256"):
        Settings.from_env({"SEED_DB_SHA256": "abc"})
    with pytest.raises(ConfigError, match="APP_PORT"):
        Settings.from_env({"APP_PORT": "70000"})
