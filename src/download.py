"""BTS downloads for `uv run python -m src.ingest --update` (P4.6). Never called at chat time.

Files are written to a folder the caller creates outside the project tree. They are untrusted
data: they are only opened as ZIP and CSV, never executed or extracted under their own names.
One request at a time, with a User-Agent, a pause before every request after the first, a
timeout, a size cap and bounded retries (network errors and 5xx answers only; a 404 means "not
published" and is never retried). Every failure raises DownloadError with a printable message.
"""
from __future__ import annotations

import csv
import io
import re
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import requests

ASPNET_STATE = ("__VIEWSTATE", "__VIEWSTATEGENERATOR", "__EVENTVALIDATION")
CHUNK_BYTES = 1 << 20


class DownloadError(Exception):
    """A download or a downloaded file failed; the message names the month and the reason."""


def session(user_agent: str) -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = user_agent
    return s


@dataclass
class PoliteClient:
    """Sequential HTTP with a pause between requests, a timeout, a size cap and bounded retries."""
    session: requests.Session
    timeout_s: float
    delay_s: float
    retries: int
    max_bytes: int
    sleep: Callable[[float], None] = time.sleep
    requests_made: int = field(default=0, init=False)

    def get_text(self, url: str) -> str:
        return self._request("GET", url, None, None)

    def download(self, url: str, dest: Path, data: dict | None = None) -> Path:
        """GET (or POST `data`) into `dest`, streamed; a partial file is removed on failure."""
        return self._request("POST" if data is not None else "GET", url, data, dest)

    def _request(self, method: str, url: str, data: dict | None, dest: Path | None):
        last = ""
        for attempt in range(self.retries + 1):
            if self.requests_made:
                self.sleep(self.delay_s * (attempt + 1))   # longer pause before a retry
            self.requests_made += 1
            try:
                response = self.session.request(method, url, data=data, timeout=self.timeout_s,
                                                stream=dest is not None, allow_redirects=True)
                if response.status_code == 404:
                    raise DownloadError("not published upstream (HTTP 404)")
                if response.status_code >= 500:
                    last = f"HTTP {response.status_code}"
                    continue
                if response.status_code != 200:
                    raise DownloadError(f"unexpected answer HTTP {response.status_code}")
                if dest is None:
                    return response.text
                self._save(response, dest)
                return dest
            except requests.RequestException as exc:
                last = type(exc).__name__   # never the URL: it may carry parameters
                if dest is not None:
                    dest.unlink(missing_ok=True)
        raise DownloadError(f"failed after {self.retries + 1} attempts ({last})")

    def _save(self, response, dest: Path) -> None:
        length = response.headers.get("Content-Length")
        if length and length.isdigit() and int(length) > self.max_bytes:
            raise DownloadError(f"refused: {int(length) / 1e6:.0f} MB is over the {self.max_bytes / 1e6:.0f} MB cap")
        written = 0
        try:
            with open(dest, "wb") as f:
                for block in response.iter_content(CHUNK_BYTES):
                    written += len(block)
                    if written > self.max_bytes:
                        raise DownloadError(f"refused: over the {self.max_bytes / 1e6:.0f} MB cap")
                    f.write(block)
        except BaseException:
            dest.unlink(missing_ok=True)
            raise


def require_zip(path: Path) -> zipfile.ZipFile:
    """The file as a ZIP; an HTML error page or a truncated file raises DownloadError."""
    if not zipfile.is_zipfile(path):
        head = path.read_bytes()[:200].lstrip().lower()
        kind = "an HTML page" if head.startswith((b"<!doctype", b"<html")) or b"<html" in head else "not a ZIP file"
        raise DownloadError(f"the download is {kind}")
    return zipfile.ZipFile(path)


def csv_header(archive: zipfile.ZipFile, member: str) -> list[str]:
    with archive.open(member) as f:
        return next(csv.reader(io.TextIOWrapper(f, encoding="utf-8-sig", newline="")), [])


def missing_columns(header: list[str], required) -> list[str]:
    present = set(header)
    return [c for c in required if c not in present]


def fetch_ontime(client: PoliteClient, url_template: str, period: str, dest_dir: Path,
                 required_columns) -> Path:
    """Download the On-Time PREZIP of `period` ('YYYY-MM') and check it holds exactly one CSV with
    every required (PREZIP) column. Returns the ZIP path, named so that the ingest glob matches."""
    year, month = map(int, period.split("-"))
    path = client.download(url_template.format(year=year, month=month), dest_dir / f"ontime_{year:04d}_{month:02d}.zip")
    with require_zip(path) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise DownloadError(f"the ZIP must hold exactly one CSV, found {names}")
        missing = missing_columns(csv_header(archive, names[0]), required_columns)
    if missing:
        raise DownloadError(f"the CSV lacks needed columns: {', '.join(missing)}")
    return path


def form_state(html: str) -> dict[str, str]:
    """The ASP.NET state fields of the T-100 download page; a missing one means the form changed."""
    state = {}
    for name in ASPNET_STATE:
        match = (re.search(rf'<input[^>]*name="{name}"[^>]*value="([^"]*)"', html)
                 or re.search(rf'<input[^>]*value="([^"]*)"[^>]*name="{name}"', html))
        if not match:
            raise DownloadError(f"the T-100 download page has no {name} field (the form changed?)")
        state[name] = match.group(1)
    return state


def fetch_t100(client: PoliteClient, page_url: str, form: dict[str, str], fields, period: str, dest_dir: Path,
               zip_csv: str, required_columns) -> Path:
    """One T-100 Segment month from the BTS form: GET the page for its state fields, POST the month
    with the configured form values and field checkboxes, check the ZIP and the CSV header, then
    write the CSV as `t100_segment_YYYY_MM.csv` (a name the ingest glob reads). Returns that path."""
    year, month = map(int, period.split("-"))
    state = form_state(client.get_text(page_url))
    data = {**state, **form, "cboYear": str(year), "cboPeriod": str(month), **{f: "on" for f in fields}}
    zip_path = client.download(page_url, dest_dir / f"t100_{year:04d}_{month:02d}.zip", data=data)
    out = dest_dir / f"t100_segment_{year:04d}_{month:02d}.csv"
    with require_zip(zip_path) as archive:
        if zip_csv not in archive.namelist():
            raise DownloadError(f"the ZIP has no {zip_csv} (found {archive.namelist()})")
        if archive.getinfo(zip_csv).file_size > client.max_bytes:
            raise DownloadError(f"refused: {zip_csv} is over the {client.max_bytes / 1e6:.0f} MB cap")
        missing = missing_columns(csv_header(archive, zip_csv), required_columns)
        if missing:
            raise DownloadError(f"the CSV lacks needed columns: {', '.join(missing)}")
        with archive.open(zip_csv) as src, open(out, "wb") as dst:
            for block in iter(lambda: src.read(CHUNK_BYTES), b""):
                dst.write(block)
    zip_path.unlink()
    return out
