"""First-pass analysis of BTS T-100 Segment (All Carriers), 2024-2025.
Usage: python3 -I scripts/analyze_t100.py  (run from project root)
"""
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
BASE = "data/raw/t100_segment/t100_segment_{y}.csv"
dtypes = {"AIRCRAFT_CONFIG": str, "CLASS": str, "AIRCRAFT_TYPE": str}
df = pd.concat([pd.read_csv(BASE.format(y=y), dtype=dtypes) for y in (2024, 2025)], ignore_index=True)

def sec(t): print("\n" + "=" * 8, t, "=" * 8)

sec("PROFILE")
print("rows", len(df), df.groupby("YEAR").size().to_dict())
print("nulls (non-zero only):", {k: int(v) for k, v in df.isna().sum().items() if v})
zero = df["DEPARTURES_PERFORMED"] == 0
print("rows with 0 departures performed:", int(zero.sum()), f"({zero.mean():.1%})",
      "| of those with passengers>0:", int((zero & (df.PASSENGERS > 0)).sum()))
print("negative values:", int((df[["DEPARTURES_PERFORMED", "SEATS", "PASSENGERS", "DISTANCE"]] < 0).sum().sum()))
print("config counts:", df.AIRCRAFT_CONFIG.value_counts().to_dict())
print("class counts:", df.CLASS.value_counts().to_dict())
print("distinct origins:", df.ORIGIN.nunique(), "| carriers:", df.UNIQUE_CARRIER.nunique())

d = df[df.DEPARTURES_PERFORMED > 0].copy()
d["pax_seat_rows"] = d.SEATS > 0

sec("CROSS-CHECK vs ArcGIS NTAD 2024 (passengers / departures by origin)")
arc = {"ANC": (2702278, 71765), "BOS": (16874194, 164312), "SFO": (17666714, 138571),
       "LAX": (26340206, 206637), "SNA": (5207539, 47578)}
g = d[d.YEAR == 2024].groupby("ORIGIN")[["PASSENGERS", "DEPARTURES_PERFORMED"]].sum()
rows = []
for k, (p, dep) in arc.items():
    rows.append((k, int(g.loc[k, "PASSENGERS"]), p, round(g.loc[k, "PASSENGERS"] / p, 3),
                 int(g.loc[k, "DEPARTURES_PERFORMED"]), dep, round(g.loc[k, "DEPARTURES_PERFORMED"] / dep, 3)))
print(pd.DataFrame(rows, columns=["origin", "pax_t100", "pax_arcgis", "ratio", "dep_t100", "dep_arcgis", "ratio"]).to_string(index=False))

sec("ANC: share of departures by distance threshold")
anc = d[d.ORIGIN == "ANC"]
out = []
for y in (2024, 2025):
    a = anc[anc.YEAR == y]
    groups = {"all": a, "passenger-config(1)": a[a.AIRCRAFT_CONFIG == "1"], "cargo-config(2)": a[a.AIRCRAFT_CONFIG == "2"],
              "combi(3)": a[a.AIRCRAFT_CONFIG == "3"]}
    for name, s in groups.items():
        tot = s.DEPARTURES_PERFORMED.sum()
        row = {"year": y, "group": name, "departures": int(tot)}
        for t in (1500, 2500, 3000):
            row[f">={t}mi"] = round(100 * s.loc[s.DISTANCE >= t, "DEPARTURES_PERFORMED"].sum() / tot, 1) if tot else None
        out.append(row)
print(pd.DataFrame(out).to_string(index=False))
a25 = anc[anc.YEAR == 2025].groupby("DEST").agg(dep=("DEPARTURES_PERFORMED", "sum"), dist=("DISTANCE", "max"),
        cargo_dep=("DEPARTURES_PERFORMED", lambda s: s[anc.loc[s.index, "AIRCRAFT_CONFIG"] == "2"].sum()))
print("\nTop ANC destinations 2025 (departures):")
print(a25.sort_values("dep", ascending=False).head(12).astype({"dep": int, "cargo_dep": int}).to_string())

sec("SFO load factor (rows with seats>0 only)")
def lf(s): return s.PASSENGERS.sum() / s.SEATS.sum()
sfo = d[(d.ORIGIN == "SFO") & d.pax_seat_rows]
print({y: round(lf(sfo[sfo.YEAR == y]) * 100, 1) for y in (2024, 2025)})
m = sfo[sfo.YEAR == 2025].groupby("MONTH").apply(lambda s: pd.Series({"seats": s.SEATS.sum(), "pax": s.PASSENGERS.sum(), "LF%": round(100 * lf(s), 1)}), include_groups=False)
print(m.astype({"seats": int, "pax": int}).to_string())
peers = d[(d.YEAR == 2025) & d.pax_seat_rows].groupby("ORIGIN").apply(lambda s: pd.Series({"seats": s.SEATS.sum(), "LF%": 100 * lf(s)}), include_groups=False)
peers = peers[peers.seats >= 5_000_000].sort_values("LF%", ascending=False)
peers["rank"] = range(1, len(peers) + 1)
print(f"\nPeers with >=5M seats in 2025: {len(peers)} airports. SFO rank by LF:", int(peers.loc["SFO", "rank"]))
print(peers.head(10).round(1).to_string())
r = sfo[sfo.YEAR == 2025].groupby("DEST").apply(lambda s: pd.Series({"seats": s.SEATS.sum(), "pax": s.PASSENGERS.sum(), "LF%": round(100 * lf(s), 1)}), include_groups=False)
r = r[r.seats >= 100_000].sort_values("LF%", ascending=False)
print("\nSFO routes (>=100k seats) with highest load factor 2025:")
print(r.head(10).astype({"seats": int, "pax": int}).to_string())

sec("NEW ENGLAND (origin states MA CT RI NH VT ME), passenger service")
ne = d[d.ORIGIN_STATE_ABR.isin(["MA", "CT", "RI", "NH", "VT", "ME"]) & d.pax_seat_rows]
t = ne.groupby(["ORIGIN", "YEAR"]).apply(lambda s: pd.Series({"dep": s.DEPARTURES_PERFORMED.sum(), "seats": s.SEATS.sum(),
        "pax": s.PASSENGERS.sum(), "LF%": 100 * lf(s)}), include_groups=False).unstack("YEAR")
res = pd.DataFrame({"pax_2025": t[("pax", 2025)], "pax_growth%": 100 * (t[("pax", 2025)] / t[("pax", 2024)] - 1),
                    "seats_growth%": 100 * (t[("seats", 2025)] / t[("seats", 2024)] - 1),
                    "LF%_2025": t[("LF%", 2025)], "dep_2025": t[("dep", 2025)]})
res = res[res.pax_2025 >= 100_000].sort_values("pax_2025", ascending=False)
print(res.round(1).to_string())

sec("LAX vs SNA (passenger service, 2025)")
for ap in ("LAX", "SNA"):
    s = d[(d.ORIGIN == ap) & (d.YEAR == 2025) & d.pax_seat_rows]
    avg_stage = (s.DISTANCE * s.DEPARTURES_PERFORMED).sum() / s.DEPARTURES_PERFORMED.sum()
    print(ap, {"dep": int(s.DEPARTURES_PERFORMED.sum()), "seats": int(s.SEATS.sum()), "pax": int(s.PASSENGERS.sum()),
               "LF%": round(100 * lf(s), 1), "avg_stage_mi": round(avg_stage), "seats_per_dep": round(s.SEATS.sum() / s.DEPARTURES_PERFORMED.sum())})
