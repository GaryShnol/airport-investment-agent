"""Make sure a usable DB exists before the app starts: `python -m src.bootstrap` (the container entrypoint).

1. DB_PATH valid (`store.db_problem`: schema version matches, every table has rows) -> nothing to do.
2. Otherwise download SEED_DB_URL (gzip), check its sha256 (SEED_DB_SHA256, else the sibling
   `.sha256` file next to it), decompress to a temp file next to DB_PATH, validate it and move it into place
   with os.replace. A bad checksum or a cut download never leaves a half-written DB: temp files are
   removed and DB_PATH is untouched. An invalid DB already there is kept as `<name>.invalid`.
3. No data -> exit 1 with the three ways to get it, no traceback.
4. Then `ingest --check` with BOOTSTRAP_CHECK_TIMEOUT_S per request; any failure = freshness unknown.

Only the URL's host and path are printed (a presigned URL's query string can hold a token).
"""
from __future__ import annotations

import dataclasses
import gzip
import hashlib
import os
import re
import shutil
import sys
import tempfile
import zlib
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import requests

from src import census, ingest, store
from src.config import ConfigError, Settings, get_settings

_CHUNK = 1024 * 1024
_SHA256 = re.compile(r"\b([0-9a-fA-F]{64})\b")


class BootstrapError(RuntimeError):
    """The DB could not be prepared; the message is for the person starting the app."""


def safe_url(url: str) -> str:
    """scheme://host[:port]/path, without credentials, query string or fragment."""
    parts = urlsplit(url)
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{parts.hostname or ''}{port}{parts.path}"


def checksum_url(url: str) -> str:
    """The sibling `.sha256` file: appended to the path, so a query string stays at the end."""
    return urlunsplit(urlsplit(url)._replace(path=urlsplit(url).path + ".sha256"))


def no_data_message(settings: Settings, reason: str) -> str:
    return (f"No usable database at {settings.db_path}: {reason}.\n"
            "Get the data in one of three ways:\n"
            "  1. Set SEED_DB_URL in .env to the seed DB release asset (see .env.example) and start again.\n"
            f"  2. Mount an existing airport.db at {settings.db_path} (docker-compose.yml: the data volume).\n"
            "  3. Put the raw files in RAW_DATA_DIR and run `uv run python -m src.ingest`.")


def expected_sha256(session, settings: Settings) -> str:
    """SEED_DB_SHA256 if set, else the first sha256 in `<SEED_DB_URL>.sha256` (sha256sum format)."""
    if settings.seed_db_sha256:
        return settings.seed_db_sha256
    url = checksum_url(settings.seed_db_url)
    try:
        response = session.get(url, timeout=settings.http_timeout_s)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise BootstrapError(f"could not get the checksum {safe_url(url)} ({type(exc).__name__}); "
                             "set SEED_DB_SHA256 or publish the .sha256 file next to the seed") from None
    match = _SHA256.search(response.text)
    if not match:
        raise BootstrapError(f"no sha256 found in {safe_url(url)}")
    return match.group(1).lower()


def download(session, settings: Settings, target: Path) -> str:
    """Stream SEED_DB_URL into `target`; return the sha256 of the bytes received."""
    url, digest, received = settings.seed_db_url, hashlib.sha256(), 0
    try:
        with session.get(url, stream=True, timeout=settings.http_timeout_s) as response:
            response.raise_for_status()
            size = response.headers.get("Content-Length")
            with open(target, "wb") as f:
                for chunk in response.iter_content(_CHUNK):
                    f.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
    except requests.RequestException as exc:
        raise BootstrapError(f"download of {safe_url(url)} failed ({type(exc).__name__})") from None
    if size is not None and size.isdigit() and int(size) != received:
        raise BootstrapError(f"download of {safe_url(url)} was cut: {received:,} of {int(size):,} bytes")
    return digest.hexdigest()


def fetch_seed(session, settings: Settings) -> None:
    """Download, verify, decompress and atomically install the seed DB at DB_PATH."""
    if not settings.seed_db_url:
        raise BootstrapError("SEED_DB_URL is not set")
    db = settings.db_path
    db.parent.mkdir(parents=True, exist_ok=True)
    expected = expected_sha256(session, settings)
    print(f"bootstrap: downloading the seed DB from {safe_url(settings.seed_db_url)}")
    # Temp files in the DB folder: os.replace is atomic only within one file system.
    with tempfile.TemporaryDirectory(dir=db.parent, prefix=".bootstrap-") as tmp:
        packed, raw = Path(tmp) / "seed.db.gz", Path(tmp) / "seed.db"
        actual = download(session, settings, packed)
        if actual != expected:
            raise BootstrapError(f"checksum mismatch for {safe_url(settings.seed_db_url)}: expected {expected}, "
                                 f"got {actual}; nothing installed")
        try:
            with gzip.open(packed, "rb") as src, open(raw, "wb") as dst:
                shutil.copyfileobj(src, dst, _CHUNK)
        except (OSError, EOFError, zlib.error) as exc:
            raise BootstrapError(f"the seed is not a valid gzip file ({type(exc).__name__})") from None
        problem = store.db_problem(raw)
        if problem:
            raise BootstrapError(f"the downloaded seed DB is not usable: {problem}")
        if db.exists():
            os.replace(db, db.with_name(db.name + ".invalid"))
        os.replace(raw, db)
    print(f"bootstrap: seed DB installed at {db}")


def check_freshness(settings: Settings) -> None:
    """`ingest --check` with a short timeout. Never fails startup: on any error freshness stays unknown."""
    try:
        ingest.check(dataclasses.replace(settings, http_timeout_s=settings.bootstrap_check_timeout_s))
    except Exception as exc:  # noqa: BLE001 - freshness is optional, the app must still start
        print(f"bootstrap: freshness check skipped ({type(exc).__name__}); freshness unknown")


def run(settings: Settings, session) -> None:
    problem = store.db_problem(settings.db_path)
    if problem is None:
        print(f"bootstrap: database OK at {settings.db_path}")
    else:
        print(f"bootstrap: {problem}")
        try:
            fetch_seed(session, settings)
        except BootstrapError as exc:
            raise BootstrapError(no_data_message(settings, str(exc))) from None
    check_freshness(settings)


def main() -> None:
    try:
        settings = get_settings()
        run(settings, census.session())
    except (BootstrapError, ConfigError) as exc:
        print(f"bootstrap: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
