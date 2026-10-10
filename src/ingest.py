"""Offline ingest: raw files in RAW_DATA_DIR -> SQLite at DB_PATH. Idempotent.

Run: `uv run python -m src.ingest`. The only module that writes to the DB.
"""
from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src import store
from src.config import get_settings, load_scoring

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
    inserted = 0
    for path in sorted(Path(t100_dir).glob(T100_GLOB)):
        df = read_t100(path)
        df["period"] = df["year"].map("{:04d}".format) + "-" + df["month"].map("{:02d}".format)
        new = df[~df["period"].isin(store.loaded_periods(conn, T100_SOURCE))]
        if new.empty:
            continue
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
        inserted += len(new)
    _update_meta(conn, T100_SOURCE)
    return inserted


def _update_meta(conn: sqlite3.Connection, source: str) -> None:
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
                     float(r.latitude_deg), float(r.longitude_deg)))
    conn.execute("DELETE FROM airports")
    conn.executemany("INSERT INTO airports (iata, name, state, region, cbsa, lat, lon) VALUES (?,?,?,?,?,?,?)", rows)
    return len(rows)


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    settings = get_settings()
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


if __name__ == "__main__":
    main()
