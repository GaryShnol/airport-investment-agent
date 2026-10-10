"""Offline ingest: raw files in RAW_DATA_DIR -> SQLite at DB_PATH. Idempotent.

Run: `uv run python -m src.ingest`. The only module that writes to the DB.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sqlite3
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src import census, scoring, store, upstream
from src.config import ScoringConfig, Settings, get_settings, load_scoring, load_sources

T100_SOURCE = "t100"
T100_GLOB = "t100*.csv"  # BTS exports; documentation.csv in the same folder is not data
T100_COLUMNS = [
    "ORIGIN", "ORIGIN_STATE_ABR", "DEST", "YEAR", "MONTH", "AIRCRAFT_CONFIG",
    "DISTANCE", "DEPARTURES_PERFORMED", "SEATS", "PASSENGERS",
]
# BTS AIRCRAFT_CONFIG codes. Any other code is treated as undocumented ("unknown"),
# so it never enters passenger metrics.
T100_CONFIGS = {"1": "pax", "2": "cargo", "3": "combi", "4": "unknown"}
T100_KEYS = ["origin", "origin_state", "dest", "year", "month", "config", "distance"]

ONTIME_SOURCE = store.ONTIME_SOURCE
ONTIME_DIR = "ontime"
ONTIME_GLOB = "**/ontime_*.zip"   # one BTS TranStats monthly download per file, read from the ZIP
ONTIME_COLUMNS = [
    "YEAR", "MONTH", "ORIGIN", "DEST", "DEP_DEL15", "TAXI_OUT", "ARR_DEL15", "CANCELLED", "DIVERTED",
    "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
]
ONTIME_CAUSES = {"CARRIER_DELAY": "carrier_min", "WEATHER_DELAY": "weather_min", "NAS_DELAY": "nas_min",
                 "SECURITY_DELAY": "security_min", "LATE_AIRCRAFT_DELAY": "late_aircraft_min"}
ONTIME_TABLE_COLUMNS = [
    "airport", "year", "month", "dep_flights", "dep_del15", "dep_cancelled", "taxi_out_sum", "taxi_out_n",
    "arr_flights", "arr_del15", *ONTIME_CAUSES.values(), "dep_del15_n",
]
ONTIME_CHUNK_ROWS = 200_000      # a month is ~540k-640k flights; read it in a few chunks

CENSUS_SOURCE = store.CENSUS_SOURCE   # ingest_log period = vintage ('2024'); meta last_period = latest vintage loaded

# Geocoder: stop after this many failures in a row (the service or the network is down; the rest is
# retried on the next run instead of waiting for a timeout per airport).
GEOCODER_MAX_CONSECUTIVE_FAILURES = 5

AIRPORTS_FILE = "airports.csv"  # OurAirports, in RAW_DATA_DIR
# US territories have their own iso_country in OurAirports; T-100 uses the same code as
# ORIGIN_STATE_ABR for PR and VI, so the country code serves as the "state".
US_TERRITORIES = {"PR", "VI", "GU", "AS", "MP"}
# A few IATA codes appear more than once (closed fields keep old codes). Keep the row with
# scheduled service, then the largest type; ties keep the first row in the file.
AIRPORT_TYPE_RANK = {"large_airport": 0, "medium_airport": 1, "small_airport": 2}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_t100(path: Path) -> pd.DataFrame:
    """Read one T-100 CSV and aggregate it to route-month-config-distance rows."""
    # All columns as strings first: config/class/type codes must not become numbers, and
    # keep_default_na=False stops codes such as "NA" from turning into NaN.
    raw = pd.read_csv(path, usecols=T100_COLUMNS, dtype=str, keep_default_na=False)
    numeric = ["DISTANCE", "DEPARTURES_PERFORMED", "SEATS", "PASSENGERS", "YEAR", "MONTH"]
    for col in numeric:
        raw[col] = pd.to_numeric(raw[col])
    raw = raw[raw["DEPARTURES_PERFORMED"] > 0]

    df = pd.DataFrame({
        "origin": raw["ORIGIN"].str.strip(),
        # Null for most foreign origins, but Canadian origins carry a province code; the peer set
        # therefore uses airports.state, not this column (docs/plan.md, Definitions).
        "origin_state": raw["ORIGIN_STATE_ABR"].str.strip().replace("", None),
        "dest": raw["DEST"].str.strip(),
        "year": raw["YEAR"].astype(int),
        "month": raw["MONTH"].astype(int),
        "config": raw["AIRCRAFT_CONFIG"].str.strip().map(T100_CONFIGS).fillna("unknown"),
        "distance": raw["DISTANCE"].round().astype(int),
        "departures": raw["DEPARTURES_PERFORMED"],
        "seats": raw["SEATS"],
        "passengers": raw["PASSENGERS"],
    })
    agg = df.groupby(T100_KEYS, dropna=False, as_index=False)[["departures", "seats", "passengers"]].sum()
    agg[["departures", "seats", "passengers"]] = agg[["departures", "seats", "passengers"]].round().astype(int)
    agg["origin_state"] = agg["origin_state"].astype(object).where(agg["origin_state"].notna(), None)
    return agg


def ingest_t100(conn: sqlite3.Connection, t100_dir: Path) -> int:
    """Load every T-100 month not yet in ingest_log. Returns the number of rows inserted.

    The unit of idempotency is (source, month): a month already loaded is skipped even if it
    appears again in another file, so reruns never duplicate rows.
    """
    inserted = sum(len(load_t100_file(conn, path)) for path in sorted(Path(t100_dir).glob(T100_GLOB)))
    update_meta(conn, T100_SOURCE)
    return inserted


def load_t100_file(conn: sqlite3.Connection, path: Path, expect_period: str | None = None) -> pd.DataFrame:
    """Insert the months of one T-100 file that are not in ingest_log yet; returns the rows inserted.

    A month already loaded is skipped, never replaced (BTS revisions of a loaded month are not
    picked up). `expect_period` ('YYYY-MM', used by `--update`) raises before any write if the
    file holds anything else.
    """
    df = read_t100(path)
    df["period"] = df["year"].map("{:04d}".format) + "-" + df["month"].map("{:02d}".format)
    if expect_period is not None and set(df["period"]) != {expect_period}:
        raise ValueError(f"{path.name}: expected only {expect_period}, found {sorted(set(df['period']))}")
    new = df[~df["period"].isin(store.loaded_periods(conn, T100_SOURCE))]
    if new.empty:
        return new
    cols = ["origin", "origin_state", "dest", "year", "month", "config", "distance",
            "departures", "seats", "passengers"]
    conn.executemany(
        f"INSERT INTO t100_route_month ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
        new[cols].itertuples(index=False, name=None),
    )
    loaded_at = _now()
    conn.executemany(
        "INSERT INTO ingest_log (source, period, rows, loaded_at) VALUES (?, ?, ?, ?)",
        [(T100_SOURCE, p, int(n), loaded_at) for p, n in new.groupby("period").size().items()],
    )
    return new


@dataclass(frozen=True)
class OntimeMonth:
    """One On-Time file aggregated per airport, plus the counts the ingest summary reports."""
    period: str                     # 'YYYY-MM'
    table: pd.DataFrame             # ONTIME_TABLE_COLUMNS, one row per airport
    flights: int                    # data rows in the file
    cancelled: int
    diverted: int
    cancelled_with_dep_delay: int   # cancelled flights that still report DEP_DEL15 (kept as reported)
    cancelled_with_taxi_out: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _ontime_csv(path: Path) -> tuple[zipfile.ZipFile, str]:
    """The ZIP and the name of its single CSV; raises for anything else (e.g. an HTML error page)."""
    if not zipfile.is_zipfile(path):
        raise ValueError(f"{path.name} is not a valid ZIP file")
    archive = zipfile.ZipFile(path)
    names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
    if len(names) != 1:
        archive.close()
        raise ValueError(f"{path.name} must hold exactly one CSV, found {names}")
    return archive, names[0]


def ontime_source_columns(rename: dict[str, str] | None) -> dict[str, str]:
    """File column -> our column for every column the aggregation reads. `rename` is the PREZIP
    mapping (config/sources.yaml); None = a TranStats download, whose names are already ours."""
    if rename is None:
        return {c: c for c in ONTIME_COLUMNS}
    wanted = {ours: src for src, ours in rename.items() if ours in ONTIME_COLUMNS}
    missing = [c for c in ONTIME_COLUMNS if c not in wanted]
    if missing:
        raise ValueError(f"the column mapping has no source for {', '.join(missing)}")
    return {wanted[c]: c for c in ONTIME_COLUMNS}


def read_ontime(path: Path, chunk_rows: int = ONTIME_CHUNK_ROWS, rename: dict[str, str] | None = None) -> OntimeMonth:
    """Stream one monthly On-Time ZIP and aggregate it per airport (docs/plan.md, On-Time sides).

    Departure side by ORIGIN: scheduled departures (incl. cancelled), cancellations, DEP_DEL15
    (sum and the number of flights reporting it), taxi-out. Arrival side by DEST: arrivals
    operated (ARR_DEL15 reported), ARR_DEL15 and the cause minutes. Empty fields are missing, not
    zero: cancelled flights mostly have no taxi-out or delay, and cause minutes exist only for
    arrivals delayed >= 15 min, so they sum as "no minutes" and never enter a count.
    `rename` reads a PREZIP file (CamelCase columns) through the mapping in config/sources.yaml.
    Raises if a needed column is missing or the file spans several months.
    """
    columns = ontime_source_columns(rename)
    source_of = {ours: src for src, ours in columns.items()}
    archive, name = _ontime_csv(path)
    dep_parts, arr_parts, periods = [], [], set()
    flights = cancelled = diverted = cancelled_dep = cancelled_taxi = 0
    with archive.open(name) as f:
        header = next(csv.reader(io.TextIOWrapper(f, encoding="utf-8-sig", newline="")), [])
        missing = [src for src in columns if src not in header]
    if missing:
        archive.close()
        raise ValueError(f"{path.name} lacks needed columns: {', '.join(missing)}")
    with archive, archive.open(name) as f:
        chunks = pd.read_csv(f, usecols=list(columns), dtype={source_of["ORIGIN"]: str, source_of["DEST"]: str},
                             keep_default_na=False, na_values=[""], chunksize=chunk_rows)
        for chunk in chunks:
            chunk = chunk.rename(columns=columns)
            periods |= set(zip(chunk["YEAR"].astype(int), chunk["MONTH"].astype(int)))
            is_cancelled = chunk["CANCELLED"] == 1
            flights += len(chunk)
            cancelled += int(is_cancelled.sum())
            diverted += int((chunk["DIVERTED"] == 1).sum())
            cancelled_dep += int((is_cancelled & chunk["DEP_DEL15"].notna()).sum())
            cancelled_taxi += int((is_cancelled & chunk["TAXI_OUT"].notna()).sum())
            dep_parts.append(chunk.groupby("ORIGIN").agg(
                dep_flights=("CANCELLED", "size"), dep_cancelled=("CANCELLED", "sum"),
                dep_del15=("DEP_DEL15", "sum"), dep_del15_n=("DEP_DEL15", "count"),
                taxi_out_sum=("TAXI_OUT", "sum"), taxi_out_n=("TAXI_OUT", "count")))
            arr_parts.append(chunk.groupby("DEST").agg(
                arr_flights=("ARR_DEL15", "count"), arr_del15=("ARR_DEL15", "sum"),
                **{col: (src, "sum") for src, col in ONTIME_CAUSES.items()}))
    if len(periods) != 1:
        raise ValueError(f"{path.name} must hold exactly one month, found {sorted(periods)}")
    (year, month), = periods

    dep = pd.concat(dep_parts).groupby(level=0).sum()
    arr = pd.concat(arr_parts).groupby(level=0).sum()
    # Outer join: an airport seen on one side only gets zeros on the other.
    table = dep.join(arr, how="outer").fillna(0).rename_axis("airport").reset_index()
    table["airport"] = table["airport"].str.strip()
    table["year"], table["month"] = year, month
    counts = ["dep_flights", "dep_cancelled", "dep_del15", "dep_del15_n", "taxi_out_n", "arr_flights", "arr_del15"]
    table[counts] = table[counts].round().astype(int)
    return OntimeMonth(f"{year:04d}-{month:02d}", table[ONTIME_TABLE_COLUMNS], flights, cancelled, diverted,
                       cancelled_dep, cancelled_taxi)


def ingest_ontime(conn: sqlite3.Connection, ontime_dir: Path) -> list[OntimeMonth]:
    """Load every On-Time file whose hash is not in ingest_log; returns the months loaded.

    A file already loaded (same sha256) is skipped without being read. A new or changed file
    replaces its month only: that month's rows and log entry are deleted, then reinserted.
    """
    loaded = [m for m in (load_ontime_file(conn, path) for path in sorted(Path(ontime_dir).glob(ONTIME_GLOB))) if m]
    update_meta(conn, ONTIME_SOURCE)
    return loaded


def load_ontime_file(conn: sqlite3.Connection, path: Path, rename: dict[str, str] | None = None,
                     expect_period: str | None = None) -> OntimeMonth | None:
    """Aggregate one On-Time ZIP and write its month; None if this exact file (sha256) is loaded.

    A new or changed local file replaces its month (`ingest` on raw files). `expect_period`
    (`--update`) raises before any write if the file holds another month, so an update only ever
    adds the month it planned. `rename`: the PREZIP column mapping (see `read_ontime`).
    """
    digest = _sha256(path)
    if conn.execute("SELECT 1 FROM ingest_log WHERE source = ? AND file_hash = ?",
                    (ONTIME_SOURCE, digest)).fetchone():
        return None
    month = read_ontime(path, rename=rename)
    if expect_period is not None and month.period != expect_period:
        raise ValueError(f"{path.name}: expected {expect_period}, found {month.period}")
    year, mon = map(int, month.period.split("-"))
    conn.execute("DELETE FROM ontime_airport_month WHERE year = ? AND month = ?", (year, mon))
    conn.execute("DELETE FROM ingest_log WHERE source = ? AND period = ?", (ONTIME_SOURCE, month.period))
    conn.executemany(
        f"INSERT INTO ontime_airport_month ({','.join(ONTIME_TABLE_COLUMNS)}) "
        f"VALUES ({','.join('?' * len(ONTIME_TABLE_COLUMNS))})",
        month.table.astype(object).itertuples(index=False, name=None),
    )
    conn.execute("INSERT INTO ingest_log (source, period, rows, loaded_at, file_hash, source_rows) "
                 "VALUES (?, ?, ?, ?, ?, ?)",
                 (ONTIME_SOURCE, month.period, len(month.table), _now(), digest, month.flights))
    return month


def ontime_row_check(conn: sqlite3.Connection, expected_rows: dict[str, int],
                     periods: list[str] | None = None) -> list[str]:
    """Flights read per reference period against the BTS counts in config/sources.yaml
    (`ontime.expected_rows`, docs/plan.md Acceptance values). Only periods listed there are checked:
    a month with no reference count (e.g. 2026) is skipped, never a failure. `periods` limits the
    check to those exact keys (the months an update loaded)."""
    lines = []
    for period, expected in expected_rows.items():
        if periods is not None and period not in periods:
            continue
        got = conn.execute("SELECT COALESCE(SUM(source_rows), 0) FROM ingest_log WHERE source = ? AND period LIKE ?",
                           (ONTIME_SOURCE, f"{period}%")).fetchone()[0]
        status = "OK" if got == expected else f"DIFFERENCE {got - expected:+,}"
        lines.append(f"ontime {period}: {got:,} flights, expected {expected:,}: {status}")
    return lines


def ingest_census(conn: sqlite3.Connection, session, settings: Settings, cfg: ScoringConfig) -> list[int]:
    """Load CBSA population for the configured vintage and growth base, skipping vintages already in
    ingest_log (a rerun makes no request). Every needed vintage is fetched before anything is
    written, so a failed request leaves the tables untouched (no partial writes). Returns the
    vintages loaded; raises census.CensusError with a printable message (never the key)."""
    loaded = store.loaded_periods(conn, CENSUS_SOURCE)
    needed = [v for v in (cfg.census_vintage, cfg.census_base_vintage) if str(v) not in loaded]
    fetched = {v: census.fetch_population(session, settings.census_api_url, cfg.census_dataset, cfg.census_variable,
                                          v, settings.census_api_key, settings.http_timeout_s) for v in needed}
    now = _now()
    for vintage, rows in fetched.items():
        conn.execute("DELETE FROM census_metro WHERE year = ?", (vintage,))
        conn.executemany("INSERT INTO census_metro (cbsa, year, population, fetched_at, name) VALUES (?, ?, ?, ?, ?)",
                         [(cbsa, vintage, population, now, name) for cbsa, name, population in rows])
        conn.execute("INSERT INTO ingest_log (source, period, rows, loaded_at) VALUES (?, ?, ?, ?)",
                     (CENSUS_SOURCE, str(vintage), len(rows), now))
    if fetched:  # meta: latest vintage loaded and the retrieval date
        update_meta(conn, CENSUS_SOURCE)
    return sorted(fetched)


@dataclass
class GeocodeSummary:
    """Peer-set airports by Geocoder outcome (IATA codes, sorted)."""
    matched: list[str] = field(default_factory=list)      # inside a metro or micro area
    no_cbsa: list[str] = field(default_factory=list)      # outside every CBSA (rural, or a territory)
    failed: list[str] = field(default_factory=list)       # request failed, not cached, retried next run
    requested: int = 0                                    # Geocoder requests made (0 on a cached rerun)

    @property
    def total(self) -> int:
        return len(self.matched) + len(self.no_cbsa) + len(self.failed)


def geocode_peer_airports(conn: sqlite3.Connection, session, settings: Settings, cfg: ScoringConfig,
                          sleep=time.sleep) -> GeocodeSummary:
    """Airport -> CBSA for peer-set airports via the Census Geocoder (coordinates from `airports`).

    A cached answer (same Geocoder vintage and coordinates) is reused without a request, including
    "no CBSA". Requests are spaced by `geocoder_delay_s`; a failed request is reported and not
    cached; after GEOCODER_MAX_CONSECUTIVE_FAILURES in a row the rest count as failed without a
    request. Finally `airports.cbsa` is refreshed from the cache."""
    window = store.analysis_window(conn)
    metrics = store.airport_metrics(conn, window, cfg.long_haul_miles)
    peer = set(metrics.loc[scoring.in_peer_set(metrics, cfg), "iata"])
    coords = conn.execute("SELECT iata, lat, lon FROM airports WHERE lat IS NOT NULL AND lon IS NOT NULL").fetchall()
    cache = store.cached_cbsa(conn, cfg.geocoder_vintage)
    summary, streak = GeocodeSummary(), 0
    for iata, lat, lon in sorted(c for c in coords if c[0] in peer):
        hit = cache.get(iata)
        if hit and (hit["lat"], hit["lon"]) == (lat, lon):
            (summary.matched if hit["cbsa"] else summary.no_cbsa).append(iata)
            continue
        if streak >= GEOCODER_MAX_CONSECUTIVE_FAILURES:
            summary.failed.append(iata)
            continue
        if summary.requested:
            sleep(settings.geocoder_delay_s)
        summary.requested += 1
        try:
            found = census.geocode(session, settings.census_geocoder_url, cfg.geocoder_benchmark,
                                   cfg.geocoder_vintage, lat, lon, settings.http_timeout_s)
        except census.CensusError:
            summary.failed.append(iata)
            streak += 1
            continue
        streak = 0
        conn.execute("INSERT OR REPLACE INTO airport_cbsa (iata, lat, lon, cbsa, cbsa_name, kind, geocoder_vintage, "
                     "geocoded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                     (iata, lat, lon, found.cbsa, found.name, found.kind, cfg.geocoder_vintage, _now()))
        (summary.matched if found.cbsa else summary.no_cbsa).append(iata)
    store.apply_cbsa(conn, cfg.geocoder_vintage)
    return summary


def update_meta(conn: sqlite3.Connection, source: str) -> None:
    last = conn.execute("SELECT MAX(period) FROM ingest_log WHERE source = ?", (source,)).fetchone()[0]
    if last is None:
        return
    conn.execute(
        "INSERT INTO meta (source, last_period, fetched_at) VALUES (?, ?, ?) "
        "ON CONFLICT(source) DO UPDATE SET last_period = excluded.last_period, fetched_at = excluded.fetched_at",
        (source, last, _now()),
    )


def _state(country: str, iso_region: str) -> str | None:
    if country == "US":
        return iso_region.split("-", 1)[1]
    return country if country in US_TERRITORIES else None


def _old_code_aliases(conn: sqlite3.Connection, airports: pd.DataFrame) -> pd.DataFrame:
    """Extra rows for T-100 codes that OurAirports has since recoded (e.g. BTS PBI = OurAirports DJT).

    A T-100 code with no IATA match is aliased only if exactly one scheduled-service airport
    lists it in `keywords`; broader keyword matching hits unrelated fields. Other unmatched
    codes (mostly BTS codes for small Alaska and seaplane points) stay unmatched.
    """
    t100_codes = {r[0] for r in conn.execute("SELECT origin FROM t100_route_month UNION SELECT dest FROM t100_route_month")}
    missing = t100_codes - set(airports["iata_code"])
    candidates = airports[airports["scheduled_service"] == "yes"]
    hits: dict[str, list[int]] = {}
    for idx, keywords in candidates["keywords"].items():
        for token in {t.strip() for t in keywords.split(",")} & missing:
            hits.setdefault(token, []).append(idx)
    aliases = []
    for code, idx in hits.items():
        if len(idx) == 1:
            row = airports.loc[idx[0]].copy()
            row["iata_code"] = code
            aliases.append(row)
    return pd.DataFrame(aliases, columns=airports.columns)


def ingest_airports(conn: sqlite3.Connection, csv_path: Path, regions: dict[str, tuple[str, ...]]) -> int:
    """Replace the airports table from OurAirports (small reference table, full refresh is idempotent).

    Keeps every country (T-100 has foreign endpoints) but only rows with an IATA code.
    Region comes from config/weights.yaml; cbsa stays null until the Census step (M4).
    """
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df["iata_code"] = df["iata_code"].str.strip()
    df = df[df["iata_code"] != ""].copy()
    df["_rank"] = (df["scheduled_service"] != "yes").astype(int) * 10 + df["type"].map(AIRPORT_TYPE_RANK).fillna(9)
    df = df.sort_values("_rank", kind="stable").drop_duplicates("iata_code", keep="first")
    df = pd.concat([df, _old_code_aliases(conn, df)], ignore_index=True)

    state_region = {state: name for name, states in regions.items() for state in states}
    rows = []
    for r in df.itertuples(index=False):
        state = _state(r.iso_country, r.iso_region)
        rows.append((r.iata_code, r.name, state, state_region.get(state), None,
                     float(r.latitude_deg), float(r.longitude_deg), r.municipality or None))
    conn.execute("DELETE FROM airports")
    conn.executemany("INSERT INTO airports (iata, name, state, region, cbsa, lat, lon, municipality) "
                     "VALUES (?,?,?,?,?,?,?,?)", rows)
    return len(rows)


def check(settings: Settings) -> None:
    """`--check`: loaded period vs upstream latest per source, stored in meta. Downloads nothing."""
    if not settings.db_path.exists():
        print(f"No database at {settings.db_path}: run `uv run python -m src.ingest` first")
        return
    with store.connect_rw(settings.db_path) as conn:
        store.init_db(conn)
        rows = upstream.run_check(conn, census.session(), settings, load_scoring(settings.weights_path))
        print(f"{'source':<8} {'loaded':<9} {'upstream':<9} freshness")
        for r in rows:
            print(f"{r.source:<8} {r.loaded:<9} {r.upstream or 'unknown':<9} {r.freshness}")
        print(f"overall: {store.overall_freshness(store.meta_rows(conn))}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="compare loaded periods with upstream (stored in meta); no downloads")
    args = parser.parse_args(argv)
    settings = get_settings()
    if args.check:
        check(settings)
        return
    with store.connect_rw(settings.db_path) as conn:
        store.init_db(conn)
        n = ingest_t100(conn, settings.raw_data_dir / "t100_segment")
        print(f"t100: {n} rows loaded")
        regions = load_scoring(settings.weights_path).regions
        n = ingest_airports(conn, settings.raw_data_dir / AIRPORTS_FILE, regions)
        print(f"airports: {n} rows")
        # Kept in t100_route_month either way; they just have no name, state or region.
        missing = store.unmatched_origins(conn)
        print(f"origins without an airports match: {len(missing)} {' '.join(missing[:20])}".rstrip())

        months = ingest_ontime(conn, settings.raw_data_dir / ONTIME_DIR)
        print(f"ontime: {len(months)} months loaded ({', '.join(m.period for m in months) or 'none new'})")
        if months:
            n, cancelled = sum(m.flights for m in months), sum(m.cancelled for m in months)
            print(f"ontime cancelled: {cancelled:,} of {n:,} flights ({cancelled / n:.2%}), "
                  f"{sum(m.diverted for m in months):,} diverted. Cancelled flights count as scheduled "
                  "departures (cancellation rate) and never as arrivals; their empty taxi-out and delay "
                  "fields are skipped, not zero. Reported anyway and kept: "
                  f"{sum(m.cancelled_with_dep_delay for m in months):,} with DEP_DEL15, "
                  f"{sum(m.cancelled_with_taxi_out for m in months):,} with taxi-out.")
        for line in ontime_row_check(conn, load_sources(settings.sources_path).ontime_expected_rows):
            print(line)
        missing = store.unmatched_ontime_airports(conn)
        print(f"ontime airports without an airports match: {len(missing)} {' '.join(missing)}".rstrip())

        cfg = load_scoring(settings.weights_path)
        http = census.session()
        if not settings.census_api_key:
            print("census: skipped, CENSUS_API_KEY is not set in .env (the Census Data API requires a key)")
        else:
            try:
                vintages = ingest_census(conn, http, settings, cfg)
                print(f"census: vintages loaded {', '.join(map(str, vintages)) or 'none new'} "
                      f"({cfg.census_dataset} {cfg.census_variable}, CBSAs)")
            except census.CensusError as exc:
                print(f"census: not loaded, nothing written: {exc}")

        geo = geocode_peer_airports(conn, http, settings, cfg)
        print(f"geocoder: {len(geo.matched)} of {geo.total} peer-set airports matched to a CBSA "
              f"({geo.requested} requests, the rest from the cache)")
        print(f"geocoder: no CBSA (outside every metro and micro area): {len(geo.no_cbsa)} {' '.join(geo.no_cbsa)}"
              .rstrip())
        if geo.failed:
            print(f"geocoder: failed, retried next run: {len(geo.failed)} {' '.join(geo.failed)}")
        print(f"census: {census.ATTRIBUTION}")


if __name__ == "__main__":
    main()
