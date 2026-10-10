"""Seed DB build: `uv run python -m src.seed build`.

Copies the schema tables (aggregates, reference tables, meta, ingest_log; never raw files) from
DB_PATH into a fresh DB stamped with the schema version, VACUUMs it and writes SEED_DB_PATH (gzip)
plus `<file>.sha256` in `sha256sum` format. The seed is published as a GitHub Release asset and
fetched by `src.bootstrap`; it is not committed. Stops, writing nothing, above SEED_MAX_MB.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import shutil
import sqlite3
import sys
import tempfile
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from src import store
from src.config import Settings, get_settings

_MB = 1024 * 1024
_CHUNK = _MB


class SeedError(RuntimeError):
    """The seed cannot be built; the message says why."""


@dataclass(frozen=True)
class SeedBuild:
    raw_bytes: int          # vacuumed SQLite file
    compressed_bytes: int
    sha256: str             # of the compressed file
    path: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(seed_path: Path) -> Path:
    """The checksum file next to the seed: airport.db.gz -> airport.db.gz.sha256."""
    return seed_path.with_name(seed_path.name + ".sha256")


def copy_tables(source: Path, target: Path) -> None:
    """Fresh DB at `target` with the current schema and the rows of every schema table in `source`."""
    with closing(sqlite3.connect(target)) as conn:
        store.init_db(conn)
        conn.execute("ATTACH DATABASE ? AS src", (str(source),))  # only read from
        for table in store.SCHEMA_TABLES:
            columns = [r[1] for r in conn.execute(f"PRAGMA main.table_info({table})")]
            present = {r[1] for r in conn.execute(f"PRAGMA src.table_info({table})")}
            if not present:
                raise SeedError(f"table {table} is missing in {source}: run `uv run python -m src.ingest` first")
            cols = ", ".join(c for c in columns if c in present)
            conn.execute(f"INSERT INTO main.{table} ({cols}) SELECT {cols} FROM src.{table}")
        conn.commit()
        conn.execute("DETACH DATABASE src")
        conn.execute("VACUUM")


def build(settings: Settings) -> SeedBuild:
    # An old schema version is fine here: the copy goes into a fresh DB with the current schema.
    problem = store.db_problem(settings.db_path, check_version=False)
    if problem:
        raise SeedError(f"cannot build the seed: {problem}")
    out = settings.seed_db_path
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        raw, packed = Path(tmp) / "seed.db", Path(tmp) / "seed.db.gz"
        copy_tables(settings.db_path, raw)
        problem = store.db_problem(raw)
        if problem:
            raise SeedError(f"the copied seed DB is not valid: {problem}")
        with open(raw, "rb") as src, open(packed, "wb") as dst_file:
            # mtime=0 and no file name in the header: the same DB gives the same bytes and checksum.
            with gzip.GzipFile(filename="", mode="wb", fileobj=dst_file, compresslevel=9, mtime=0) as dst:
                shutil.copyfileobj(src, dst, _CHUNK)
        result = SeedBuild(raw.stat().st_size, packed.stat().st_size, sha256_file(packed), out)
        if result.compressed_bytes > settings.seed_max_mb * _MB:
            raise SeedError(f"compressed seed is {result.compressed_bytes / _MB:.1f} MB, over SEED_MAX_MB "
                            f"({settings.seed_max_mb:g} MB); nothing written")
        os.replace(packed, out)
    sha256_path(out).write_text(f"{result.sha256}  {out.name}\n", encoding="utf-8", newline="\n")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build"], help="build the gzipped seed DB from DB_PATH")
    parser.parse_args(argv)
    settings = get_settings()
    try:
        result = build(settings)
    except SeedError as exc:
        print(f"seed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    print(f"seed: {result.path} ({result.compressed_bytes / _MB:.1f} MB compressed, "
          f"{result.raw_bytes / _MB:.1f} MB raw after VACUUM)")
    print(f"seed: sha256 {result.sha256} -> {sha256_path(result.path)}")


if __name__ == "__main__":
    main()
