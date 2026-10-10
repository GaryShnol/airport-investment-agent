"""Seed DB build (`src.seed`): copy of the schema tables, VACUUM, gzip, sha256, size limit."""
import gzip
import hashlib
import sqlite3
from contextlib import closing

import pytest

from src import seed, store
from src.config import Settings
from tests.conftest import full_db


def settings_for(tmp_path, **env):
    return Settings.from_env({"DB_PATH": str(tmp_path / "airport.db"),
                              "SEED_DB_PATH": str(tmp_path / "seed" / "airport.db.gz"), **env})


def test_seed_file_name_does_not_clash_with_build_temp_files(tmp_path):
    """Regression: SEED_DB_PATH named airport.db once truncated the raw copy while compressing it."""
    s = settings_for(tmp_path, SEED_DB_PATH=str(tmp_path / "out" / "airport.db"))
    full_db(s.db_path)
    seed.build(s)
    raw = tmp_path / "check.db"
    raw.write_bytes(gzip.decompress(s.seed_db_path.read_bytes()))
    assert store.db_problem(raw) is None


def test_build_writes_gzip_and_checksum(tmp_path):
    s = settings_for(tmp_path)
    full_db(s.db_path)
    with closing(sqlite3.connect(s.db_path)) as c:
        c.execute("CREATE TABLE scratch (x)")   # not a schema table: must not reach the seed
        c.execute("PRAGMA user_version = 0")     # an older DB still builds a current-version seed
        c.commit()

    result = seed.build(s)

    packed = s.seed_db_path.read_bytes()
    assert result.compressed_bytes == len(packed) and result.sha256 == hashlib.sha256(packed).hexdigest()
    assert seed.sha256_path(s.seed_db_path).read_text() == f"{result.sha256}  airport.db.gz\n"
    raw = tmp_path / "unpacked.db"
    raw.write_bytes(gzip.decompress(packed))
    assert result.raw_bytes == raw.stat().st_size
    assert store.db_problem(raw) is None
    with closing(sqlite3.connect(raw)) as c:
        tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert tables == set(store.SCHEMA_TABLES)
        for table in store.SCHEMA_TABLES:
            with closing(sqlite3.connect(s.db_path)) as src:
                assert c.execute(f"SELECT COUNT(*) FROM {table}").fetchone() == \
                    src.execute(f"SELECT COUNT(*) FROM {table}").fetchone()


def test_build_is_reproducible(tmp_path):
    s = settings_for(tmp_path)
    full_db(s.db_path)
    assert seed.build(s).sha256 == seed.build(s).sha256


def test_build_stops_over_the_size_limit(tmp_path):
    s = settings_for(tmp_path, SEED_MAX_MB="0.000001")
    full_db(s.db_path)
    with pytest.raises(seed.SeedError, match="over SEED_MAX_MB"):
        seed.build(s)
    assert not s.seed_db_path.exists() and not seed.sha256_path(s.seed_db_path).exists()
    assert list(s.seed_db_path.parent.iterdir()) == []   # no temp files left


def test_build_refuses_an_incomplete_db(tmp_path):
    s = settings_for(tmp_path)
    with store.connect_rw(s.db_path) as c:
        store.init_db(c)
    with pytest.raises(seed.SeedError, match="empty"):
        seed.build(s)


def test_main_reports_errors_without_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(seed, "get_settings", lambda: settings_for(tmp_path))
    with pytest.raises(SystemExit) as exc:
        seed.main(["build"])
    assert exc.value.code == 1 and "no database" in capsys.readouterr().err


def test_db_problem(tmp_path):
    path = full_db(tmp_path / "a.db")
    assert store.db_problem(path) is None
    with closing(sqlite3.connect(path)) as c:
        c.execute(f"PRAGMA user_version = {store.SCHEMA_VERSION + 1}")
    assert "schema version" in store.db_problem(path)
    assert store.db_problem(path, check_version=False) is None
    (tmp_path / "junk.db").write_bytes(b"not sqlite at all" * 100)
    assert "not a usable SQLite database" in store.db_problem(tmp_path / "junk.db")
    assert "no database" in store.db_problem(tmp_path / "missing.db")
