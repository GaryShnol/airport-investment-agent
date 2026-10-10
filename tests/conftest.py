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


# --- small analysis DB for tools and agent tests --------------------------------------------

AIRPORTS = [  # iata, name, state, region, municipality
    ("LAX", "Los Angeles International Airport", "CA", None, "Los Angeles"),
    ("BUR", "Hollywood Burbank Airport", "CA", None, "Burbank"),
    ("LGB", "Long Beach Airport", "CA", None, "Long Beach"),
    ("ONT", "Ontario International Airport", "CA", None, "Ontario"),
    ("SNA", "John Wayne Orange County Airport", "CA", None, "Santa Ana"),
    ("ANC", "Ted Stevens Anchorage International Airport", "AK", None, "Anchorage"),
    ("MRI", "Merrill Field", "AK", None, "Anchorage"),
    ("ORD", "Chicago O'Hare International Airport", "IL", None, "Chicago"),
    ("MDW", "Chicago Midway International Airport", "IL", None, "Chicago"),
    ("BOS", "Boston Logan International Airport", "MA", "new_england", "Boston"),
    ("PWM", "Portland International Jetport", "ME", "new_england", "Portland"),
    ("BTV", "Burlington International Airport", "VT", "new_england", "Burlington"),
    ("HYA", "Cape Cod Gateway Airport", "MA", "new_england", "Hyannis"),
]


def _routes():
    """(origin, dest, config, distance, departures, seats, passengers) per month, both years."""
    rows = []
    for year, growth in ((2024, 1.0), (2025, 1.1)):
        for month in range(1, 13):
            for origin, dest, config, dist, dep, seats, pax in [
                ("ANC", "SEA", "pax", 1448, 60, 9000, 7000),
                ("ANC", "ORD", "pax", 2846, 10, 2000, 1600),     # passenger long haul
                ("ANC", "ORD", "cargo", 2846, 50, 0, 0),         # cargo long haul
                ("ANC", "HNL", "combi", 2781, 2, 200, 150),
                ("BOS", "LAX", "pax", 2611, 100, 18000, 15000),
                ("BOS", "PWM", "pax", 98, 50, 5000, 4000),
                ("PWM", "BOS", "pax", 98, 30, 3000, 2600),
                ("BTV", "BOS", "pax", 189, 20, 2000, 1500),
                ("HYA", "BOS", "pax", 60, 2, 18, 5),             # below the peer-set floor
                ("LAX", "BOS", "pax", 2611, 100, 18000, 15000),
                ("SNA", "BOS", "pax", 2600, 10, 1800, 1500),
                ("ORD", "BOS", "pax", 867, 80, 12000, 10000),
                ("MDW", "BOS", "pax", 860, 20, 3000, 2500),
            ]:
                rows.append((origin, None, dest, year, month, config, dist, dep, seats, round(pax * growth)))
    return rows


def seed_analysis_db(conn):
    """Two full years (2024-2025) of T-100 for a handful of airports; MRI has no service."""
    store.init_db(conn)
    conn.executemany("INSERT INTO airports (iata, name, state, region, municipality) VALUES (?,?,?,?,?)", AIRPORTS)
    conn.executemany("INSERT INTO t100_route_month VALUES (?,?,?,?,?,?,?,?,?,?)", _routes())
    conn.executemany("INSERT INTO ingest_log (source, period, rows, loaded_at) VALUES ('t100', ?, 1, 'now')",
                     [(f"{y}-{m:02d}",) for y in (2024, 2025) for m in range(1, 13)])
    conn.execute("INSERT INTO meta (source, last_period) VALUES ('t100', '2025-12')")
    conn.commit()


# On-Time per airport and month: scheduled departures, delayed >= 15, mean taxi-out, NAS minutes
# per arrival, cancelled. BTV is under min_ontime_flights (1,000/yr); HYA, ANC, MDW have none.
ONTIME = {
    "LAX": (1000, 250, 18.0, 2.0, 10),
    "SNA": (200, 40, 16.0, 1.5, 2),
    "ORD": (900, 270, 20.0, 4.0, 30),
    "BOS": (800, 200, 21.0, 7.0, 20),
    "PWM": (150, 27, 14.0, 1.0, 3),
    "BTV": (50, 10, 12.0, 1.0, 1),
}


def seed_ontime(conn):
    """12 months of 2025 On-Time rows built from ONTIME (counts consistent with the rates)."""
    rows = []
    for airport, (dep, del15, taxi, nas, cancelled) in ONTIME.items():
        flown = dep - cancelled
        for month in range(1, 13):
            rows.append((airport, 2025, month, dep, del15, flown, cancelled, taxi * flown, flown, flown,
                         del15, 0.0, 0.0, nas * flown, 0.0, 0.0))
    conn.executemany("INSERT INTO ontime_airport_month (airport, year, month, dep_flights, dep_del15, dep_del15_n, "
                     "dep_cancelled, taxi_out_sum, taxi_out_n, arr_flights, arr_del15, carrier_min, weather_min, "
                     "nas_min, security_min, late_aircraft_min) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.executemany("INSERT INTO ingest_log (source, period, rows, loaded_at) VALUES ('ontime', ?, 1, 'now')",
                     [(f"2025-{m:02d}",) for m in range(1, 13)])
    conn.execute("INSERT INTO meta (source, last_period) VALUES ('ontime', '2025-12')")
    conn.commit()


def seed_sfo(conn):
    """SFO for explain_demand_gap: HNL is full in June-August (950 of 1,000 seats, else 700), LAX
    flat at 1,500 of 2,000; 2024 is lower (growth). On-Time with every delay cause (minutes/month:
    carrier 100, weather 20, NAS 300, security 0, late aircraft 180)."""
    conn.execute("INSERT INTO airports (iata, name, state, region, municipality) VALUES "
                 "('SFO', 'San Francisco International Airport', 'CA', NULL, 'San Francisco')")
    rows = []
    for year, summer, other, lax in ((2024, 800, 600, 1400), (2025, 950, 700, 1500)):
        for month in range(1, 13):
            rows += [("SFO", None, "HNL", year, month, "pax", 2398, 5, 1000, summer if month in (6, 7, 8) else other),
                     ("SFO", None, "LAX", year, month, "pax", 337, 10, 2000, lax)]
    conn.executemany("INSERT INTO t100_route_month VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    conn.executemany("INSERT INTO ontime_airport_month (airport, year, month, dep_flights, dep_del15, dep_del15_n, "
                     "dep_cancelled, taxi_out_sum, taxi_out_n, arr_flights, arr_del15, carrier_min, weather_min, "
                     "nas_min, security_min, late_aircraft_min) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     [("SFO", 2025, m, 500, 100, 495, 5, 20.0 * 495, 495, 495, 90, 100.0, 20.0, 300.0, 0.0, 180.0)
                      for m in range(1, 13)])
    conn.commit()


@pytest.fixture
def scoring_cfg():
    """Real config/weights.yaml with a passenger floor that fits the tiny dataset."""
    from dataclasses import replace

    from src.config import ROOT, load_scoring
    return replace(load_scoring(ROOT / "config" / "weights.yaml"), min_passengers_per_year=1000)


# CBSA per airport and metro population (2024, 2019). HYA and BTV lie outside every CBSA here.
CENSUS = {"BOS": ("14460", "Boston-Cambridge-Newton, MA-NH Metro Area", 4_900_000, 4_800_000),
          "PWM": ("38860", "Portland-South Portland, ME Metro Area", 560_000, 530_000),
          "LAX": ("31080", "Los Angeles-Long Beach-Anaheim, CA Metro Area", 13_000_000, 13_200_000),
          "SNA": ("31080", "Los Angeles-Long Beach-Anaheim, CA Metro Area", 13_000_000, 13_200_000),
          "ANC": ("11260", "Anchorage, AK Metro Area", 400_000, 400_000),
          "ORD": ("16980", "Chicago-Naperville-Elgin, IL-IN Metro Area", 9_400_000, 9_500_000),
          "MDW": ("16980", "Chicago-Naperville-Elgin, IL-IN Metro Area", 9_400_000, 9_500_000)}


def seed_census(conn, vintage=2024, base=2019):
    """airports.cbsa as the Geocoder step leaves it, census_metro for both vintages, ingest_log."""
    conn.executemany("UPDATE airports SET cbsa = ? WHERE iata = ?", [(c[0], i) for i, c in CENSUS.items()])
    metros = {c[0]: c for c in CENSUS.values()}
    conn.executemany("INSERT INTO census_metro (cbsa, year, population, fetched_at, name) VALUES (?,?,?,?,?)",
                     [(cbsa, year, pop, "now", name) for cbsa, name, now_pop, base_pop in metros.values()
                      for year, pop in ((vintage, now_pop), (base, base_pop))])
    conn.executemany("INSERT INTO ingest_log (source, period, rows, loaded_at) VALUES ('census', ?, 1, 'now')",
                     [(str(vintage),), (str(base),)])
    conn.execute("INSERT INTO meta (source, last_period, fetched_at) VALUES ('census', ?, 'now')", (str(vintage),))
    conn.commit()
