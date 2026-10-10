"""Shared fixtures: tiny synthetic T-100 files and an empty DB in tmp_path (never data/raw)."""
import pytest

from src import store

T100_HEADER = (
    "DEPARTURES_PERFORMED,SEATS,PASSENGERS,FREIGHT,MAIL,DISTANCE,AIR_TIME,UNIQUE_CARRIER,"
    "UNIQUE_CARRIER_NAME,ORIGIN_AIRPORT_ID,ORIGIN,ORIGIN_STATE_ABR,DEST_AIRPORT_ID,DEST,"
    "AIRCRAFT_TYPE,AIRCRAFT_CONFIG,YEAR,MONTH,CLASS\n"
)


def t100_row(dep, seats, pax, dist, origin, dest, config, year, month, state="AK", carrier="AS"):
    return (
        f'{dep},{seats},{pax},0.00,0.00,{dist},0.00,{carrier},"Carrier, Inc",1,{origin},{state},'
        f"2,{dest},035,{config},{year},{month},F\n"
    )


@pytest.fixture
def t100_dir(tmp_path):
    d = tmp_path / "raw" / "t100_segment"
    d.mkdir(parents=True)
    rows = [
        # same route/month/config/distance from two carriers -> one aggregated row
        t100_row("10.00", "1000.00", "800.00", "261.00", "ANC", "FAI", "1", 2025, 1),
        t100_row("5.00", "500.00", "450.00", "261.00", "ANC", "FAI", "1", 2025, 1, carrier="DL"),
        t100_row("7.00", "0.00", "0.00", "2846.00", "ANC", "ORD", "2", 2025, 1),
        t100_row("2.00", "200.00", "150.00", "1448.00", "ANC", "SEA", "3", 2025, 1),
        t100_row("1.00", "0.00", "0.00", "1448.00", "ANC", "SEA", "4", 2025, 2),
        # zero departures -> dropped
        t100_row("0.00", "0.00", "0.00", "0.00", "ANC", "ANC", "1", 2025, 2),
        # foreign origin: no state
        t100_row("3.00", "300.00", "200.00", "2000.00", "YVR", "ANC", "1", 2025, 2, state=""),
    ]
    (d / "t100_2025.csv").write_text(T100_HEADER + "".join(rows), encoding="utf-8")
    return d


@pytest.fixture
def conn(tmp_path):
    with store.connect_rw(tmp_path / "test.db") as c:
        store.init_db(c)
        yield c
