"""
audit_data.py
=============
Audit kualitas dataset lap F1 (Bagian A tugas: Data audit).
Script ini HANYA mendeteksi & melaporkan anomali; tidak menghapus data.
Keputusan treatment (buang / imputasi / tandai) dibuat dan dijelaskan di notebook.

Jalankan setelah fastf1_download.py selesai:
    python audit_data.py
Output: laporan di terminal + file CSV di data/audit/
"""

import os
import sys
import numpy as np
import pandas as pd

# Default: data 2021-2025. Untuk audit lain: python audit_data.py ./data/f1_laps_2026.csv
DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "./data/f1_laps_all_2021_2025.csv"
LOG_PATH = "./data/download_log.csv"
OUT_DIR = "./data/audit"
os.makedirs(OUT_DIR, exist_ok=True)

DRY = {"SOFT", "MEDIUM", "HARD"}
RACE = ["Season", "RaceName"]
RACE_DRV = RACE + ["Driver"]


def section(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def to_bool(s):
    return s.astype(str).str.strip().str.lower().map({"true": True, "false": False})


def save(df, name):
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False)
    return path


df = pd.read_csv(DATA_PATH, dtype={"TrackStatus": str, "DriverNumber": str})

# ---------------------------------------------------------------
section("1. KELENGKAPAN DOWNLOAD (dari download_log.csv)")
if os.path.exists(LOG_PATH):
    log = pd.read_csv(LOG_PATH)
    print(log.groupby(["Season", "Status"]).size().unstack(fill_value=0))
    bad = log[log["Status"] != "OK"]
    if len(bad):
        print("\nRace TIDAK berhasil:")
        print(bad[["Season", "Round", "RaceName", "Status", "Message"]].to_string(index=False))
        save(bad, "missing_races.csv")
    else:
        print("Semua race pada kalender berhasil diunduh.")
else:
    print("download_log.csv tidak ditemukan; lewati.")

# ---------------------------------------------------------------
section("2. RINGKASAN DATASET")
print(f"Jumlah season   : {df['Season'].nunique()} {sorted(df['Season'].unique())}")
print(f"Jumlah race     : {df.groupby(RACE).ngroups}")
print(f"Jumlah driver   : {df['Driver'].nunique()}")
print(f"Jumlah team     : {df['Team'].nunique()}")
print(f"Compound        : {sorted(df['Compound'].dropna().unique())}")
print(f"Total baris lap : {len(df)}")
print(f"Lap dgn LapTime : {df['LapTimeSeconds'].notna().sum()}")
print("\nRace per season:")
print(df.groupby("Season").apply(lambda g: g.groupby("RaceName").ngroups).to_string())

# ---------------------------------------------------------------
section("3. MISSING VALUES PER KOLOM")
miss = pd.DataFrame({"missing": df.isna().sum(), "persen": (df.isna().mean() * 100).round(2)})
print(miss[miss["missing"] > 0].sort_values("persen", ascending=False).to_string())

# ---------------------------------------------------------------
section("4. DETEKSI ANOMALI")

# a) duplikat
dup = df[df.duplicated(RACE_DRV + ["LapNumber"], keep=False)]
print(f"a) Baris duplikat (race, driver, lap): {len(dup)}")
if len(dup):
    save(dup, "anomaly_duplicates.csv")

# b) compound tidak valid / tidak dikenal
comp_counts = df["Compound"].value_counts(dropna=False)
print("\nb) Distribusi Compound:")
print(comp_counts.to_string())
odd_comp = df[~df["Compound"].isin(DRY | {"INTERMEDIATE", "WET"})]
print(f"   Baris dengan compound non-standar/kosong: {len(odd_comp)}")
if len(odd_comp):
    save(odd_comp, "anomaly_compound.csv")

# c) LapTime kosong / <= 0
bad_lt = df[df["LapTimeSeconds"].isna() | (df["LapTimeSeconds"] <= 0)]
print(f"\nc) LapTime kosong atau <= 0: {len(bad_lt)} ({len(bad_lt) / len(df):.1%})")

# d) nomor lap yang hilang per driver per race
g = df.groupby(RACE_DRV)["LapNumber"].agg(max_lap="max", n_rows="nunique").reset_index()
g["missing_laps"] = g["max_lap"] - g["n_rows"]
gaps = g[g["missing_laps"] > 0]
print(f"\nd) Kombinasi driver-race dengan nomor lap hilang: {len(gaps)}")
if len(gaps):
    save(gaps, "anomaly_lap_gaps.csv")

# e) TyreLife turun di dalam satu stint (seharusnya naik +1 per lap)
d2 = df.sort_values(RACE_DRV + ["Stint", "LapNumber"]).copy()
d2["tyre_diff"] = d2.groupby(RACE_DRV + ["Stint"])["TyreLife"].diff()
tl_bad = d2[d2["tyre_diff"] < 0]
print(f"\ne) TyreLife turun di dalam stint yang sama: {len(tl_bad)}")
if len(tl_bad):
    save(tl_bad, "anomaly_tyrelife.csv")

# f) jumlah sektor vs LapTime
s_cols = ["Sector1TimeSeconds", "Sector2TimeSeconds", "Sector3TimeSeconds"]
if all(c in df.columns for c in s_cols):
    ssum = df[s_cols].sum(axis=1, min_count=3)
    mism = df[(ssum - df["LapTimeSeconds"]).abs() > 0.5]
    print(f"\nf) |Sektor1+2+3 - LapTime| > 0.5 detik: {len(mism)}")
    if len(mism):
        save(mism, "anomaly_sector_mismatch.csv")

# g) outlier lap time relatif terhadap median race
med = df.groupby(RACE)["LapTimeSeconds"].transform("median")
ratio = df["LapTimeSeconds"] / med
out_hi = df[ratio > 1.3]
out_lo = df[ratio < 0.85]
print(f"\ng) Outlier: > 1.3x median race = {len(out_hi)} (umumnya pit/SC/red flag), "
      f"< 0.85x median = {len(out_lo)} (mencurigakan)")
if len(out_lo):
    save(out_lo, "anomaly_too_fast.csv")

# h) track status
ts = df["TrackStatus"].dropna()
print(f"\nh) Lap dengan TrackStatus != '1' (bukan green flag): {(ts != '1').sum()} ({(ts != '1').mean():.1%})")
print("   Frekuensi kode (1=green, 2=yellow, 4=SC, 5=red, 6/7=VSC):")
print(ts.apply(list).explode().value_counts().to_string())

# i) flag bawaan FastF1
if "IsAccurate" in df.columns:
    acc = to_bool(df["IsAccurate"])
    print(f"\ni) IsAccurate = False: {(acc == False).sum()}")
if "Deleted" in df.columns:
    dele = to_bool(df["Deleted"])
    print(f"   Deleted = True    : {(dele == True).sum()}")
pit = df["PitInTime"].notna() | df["PitOutTime"].notna()
print(f"   Lap in/out pit    : {pit.sum()}")

# j) jumlah driver per race & konsistensi jumlah lap
race_sum = df.groupby(RACE).agg(
    drivers=("Driver", "nunique"),
    laps=("LapNumber", "size"),
    max_lap=("LapNumber", "max"),
    valid_laptime=("LapTimeSeconds", lambda s: s.notna().sum()),
    pct_non_green=("TrackStatus", lambda s: (s.dropna() != "1").mean()),
    has_wet=("Compound", lambda s: s.isin(["INTERMEDIATE", "WET"]).any()),
).reset_index()
print(f"\nj) Race dengan jumlah driver di luar 18-22: "
      f"{len(race_sum[(race_sum['drivers'] < 18) | (race_sum['drivers'] > 22)])}")
print(f"   Race dengan compound INTERMEDIATE/WET (wet race): {race_sum['has_wet'].sum()}")
save(race_sum, "race_summary.csv")

# k) sanity check pace per compound (dry, lap hijau)
clean_pace = df[df["Compound"].isin(DRY) & (df["TrackStatus"] == "1") & ~pit]
print("\nk) Median LapTime (detik) per compound, lap hijau non-pit (hanya sanity check):")
print(clean_pace.groupby("Compound")["LapTimeSeconds"].median().round(3).to_string())

# ---------------------------------------------------------------
section("5. FUNNEL 'CLEAN RACING LAP' (kandidat definisi untuk pemodelan)")
mask = pd.Series(True, index=df.index)
steps = [("Total lap", mask.copy())]


def add(name, cond):
    global mask
    mask = mask & cond.fillna(False)
    steps.append((name, mask.copy()))


add("LapTime valid (> 0)", df["LapTimeSeconds"].notna() & (df["LapTimeSeconds"] > 0))
if "IsAccurate" in df.columns:
    add("IsAccurate = True", to_bool(df["IsAccurate"]))
if "Deleted" in df.columns:
    add("Bukan lap Deleted", ~to_bool(df["Deleted"]).fillna(False))
add("Bukan pit in/out lap", ~pit)
add("TrackStatus = '1' (green flag)", df["TrackStatus"] == "1")
add("Compound dry (S/M/H)", df["Compound"].isin(DRY))
add("Bukan lap 1 (standing start)", df["LapNumber"] > 1)
add("TyreLife tersedia", df["TyreLife"].notna())

funnel = pd.DataFrame([(n, int(m.sum())) for n, m in steps], columns=["Filter", "Sisa lap"])
funnel["% dari total"] = (funnel["Sisa lap"] / len(df) * 100).round(1)
print(funnel.to_string(index=False))
save(funnel, "clean_lap_funnel.csv")

print(f"\nSelesai. File detail ada di: {OUT_DIR}/")