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
from src.config import get_settings

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
        # Null for foreign origins; the peer set needs a US state (docs/plan.md, Definitions).
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


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    settings = get_settings()
    with store.connect_rw(settings.db_path) as conn:
        store.init_db(conn)
        n = ingest_t100(conn, settings.raw_data_dir / "t100_segment")
        print(f"t100: {n} rows loaded")


if __name__ == "__main__":
    main()
