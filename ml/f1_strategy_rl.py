# %% [markdown]
# # F1 Race Strategy Optimization: FastF1 + Supervised Learning + RL
# ## Bagian A. Data Audit & Exploratory Analysis
#
# **Data provenance**
# - Sumber utama: data timing Formula 1 (F1 Live Timing API) yang diakses lewat library **FastF1** (tidak resmi, tidak berafiliasi dengan F1).
# - Dataset turunan dibuat oleh `fastf1_download.py` (`fastf1.get_session(season, round, "R")`, `session.load()`, `session.laps`), kolom `Season`, `Round`, `RaceName`, `RaceLaps` dan data cuaca ditambahkan oleh script.
# - Rincian status unduhan per race: `data/download_log.csv`.
# - Tanggal unduh: *(isi)*  |  Versi FastF1: lihat sel di bawah.

# %%
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)

DATA_PATH = "./data/f1_laps_all_2021_2025.csv"
CLEAN_PATH = "./data/laps_clean.csv"
FIG_DIR = "./figures"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)

DRY = ["SOFT", "MEDIUM", "HARD"]
REQUIRE_IS_ACCURATE = True   # IsAccurate = sinkronisasi waktu lap, BUKAN akurasi lap time; lihat sel sensitivitas
WET_RACE_THRESHOLD = 0.05    # race dianggap "wet" jika >= 5% lap memakai INTERMEDIATE/WET (asumsi, bisa diuji)
COMPOUND_COLORS = {"SOFT": "#e10600", "MEDIUM": "#f5c400", "HARD": "#9a9a9a",
                   "INTERMEDIATE": "#3fa34d", "WET": "#1f6fd0"}

try:
    import fastf1
    print("FastF1 version:", fastf1.__version__)
except Exception:
    print("FastF1 tidak terpasang di environment ini (tidak masalah untuk analisis).")

# %%
df = pd.read_csv(DATA_PATH, dtype={"TrackStatus": str, "DriverNumber": str})


def to_bool(s):
    return s.astype(str).str.strip().str.lower().map({"true": True, "false": False})


for c in ["IsAccurate", "Deleted", "FreshTyre", "IsPersonalBest"]:
    if c in df.columns:
        df[c] = to_bool(df[c])

df.info()
df.head()

# %% [markdown]
# ### A.1 Ringkasan dataset

# %%
print(f"Season  : {df['Season'].nunique()} {sorted(df['Season'].unique())}")
print(f"Race    : {df.groupby(['Season', 'RaceName']).ngroups}")
print(f"Driver  : {df['Driver'].nunique()}")
print(f"Team    : {df['Team'].nunique()}")
print(f"Compound: {sorted(df['Compound'].dropna().unique())}")
print(f"Total baris lap        : {len(df):,}")
print(f"Lap dengan LapTime ada : {df['LapTimeSeconds'].notna().sum():,}")

per_season = df.groupby("Season").agg(
    race=("RaceName", "nunique"), driver=("Driver", "nunique"),
    team=("Team", "nunique"), lap=("LapNumber", "size"))
display(per_season)

# Nama team berbeda antar musim (rebranding). Perlu pemetaan sebelum dipakai sebagai fitur kategori.
display(df.groupby("Season")["Team"].apply(lambda s: sorted(s.dropna().unique())).to_frame("team"))

# %% [markdown]
# ### A.2 Konversi waktu, flag anomali, dan definisi *clean racing lap*
#
# `LapTimeSeconds` dan `Sector*Seconds` sudah dikonversi dari `timedelta` ke detik oleh script unduhan.
# Notebook ini **menandai** (flag) setiap kondisi bermasalah terlebih dahulu, baru kemudian memutuskan treatment-nya.

# %%
# --- Flag dasar ---
df["RaceLaps"] = df.groupby(["Season", "RaceName"])["LapNumber"].transform("max") \
    if "RaceLaps" not in df.columns else df["RaceLaps"]

df["is_missing_laptime"] = df["LapTimeSeconds"].isna() | (df["LapTimeSeconds"] <= 0)
df["is_inaccurate"] = ~df["IsAccurate"].fillna(False).astype(bool) if "IsAccurate" in df.columns else False
df["is_deleted"] = df["Deleted"].fillna(False).astype(bool) if "Deleted" in df.columns else False
if "Deleted" not in df.columns or df["Deleted"].isna().all():
    print("CATATAN: kolom Deleted kosong 100% (race control messages tidak dimuat saat download). "
          "Deteksi lap deleted TIDAK tersedia; ini dicatat sebagai limitation.")
df["is_pit_lap"] = df["PitInTime"].notna() | df["PitOutTime"].notna()
df["is_green"] = df["TrackStatus"] == "1"
df["is_sc_vsc_red"] = df["TrackStatus"].fillna("").str.contains("[4567]")   # 4=SC, 5=red, 6/7=VSC

# --- Outlier pace: rasio terhadap median race (lap hijau, dry, bukan pit) ---
base = (~df["is_missing_laptime"]) & (~df["is_pit_lap"]) & df["is_green"] & df["Compound"].isin(DRY)
ref = (df[base].groupby(["Season", "RaceName"])["LapTimeSeconds"].median().rename("race_median").reset_index())
df = df.merge(ref, on=["Season", "RaceName"], how="left")
df["pace_ratio"] = df["LapTimeSeconds"] / df["race_median"]
df["pace_delta"] = df["LapTimeSeconds"] - df["race_median"]      # detik relatif terhadap median race
df["is_outlier"] = (df["pace_ratio"] > 1.15) | (df["pace_ratio"] < 0.90)
df["laps_remaining"] = df["RaceLaps"] - df["LapNumber"]

# --- Wet race: proporsi lap INTERMEDIATE/WET per race ---
wet = (df.groupby(["Season", "RaceName"])["Compound"]
         .apply(lambda s: s.isin(["INTERMEDIATE", "WET"]).mean()).rename("wet_share").reset_index())
df = df.merge(wet, on=["Season", "RaceName"], how="left")
df["is_wet_race"] = df["wet_share"] >= WET_RACE_THRESHOLD
print(f"Race dengan lap INTERMEDIATE/WET: {(wet['wet_share'] > 0).sum()} | "
      f"dianggap wet race (>= {WET_RACE_THRESHOLD:.0%}): {wet[wet['wet_share'] >= WET_RACE_THRESHOLD].shape[0]}")
display(wet[wet["wet_share"] > 0].sort_values("wet_share", ascending=False).round(3))

flags = ["is_missing_laptime", "is_inaccurate", "is_deleted", "is_pit_lap",
         "is_sc_vsc_red", "is_outlier"]
display(pd.DataFrame({"jumlah": df[flags].sum(), "persen": (df[flags].mean() * 100).round(2)}))

# %%
# --- Funnel: dari seluruh lap ke clean racing lap ---
steps = [
    ("Total lap", pd.Series(True, index=df.index)),
    ("LapTime valid (> 0)", ~df["is_missing_laptime"]),
    ("IsAccurate = True" if REQUIRE_IS_ACCURATE else "IsAccurate (tidak dipakai)",
     ~df["is_inaccurate"] if REQUIRE_IS_ACCURATE else pd.Series(True, index=df.index)),
    ("Bukan lap deleted (kolom kosong, tidak berefek)", ~df["is_deleted"]),
    ("Bukan pit in/out lap", ~df["is_pit_lap"]),
    ("Green flag (TrackStatus = '1')", df["is_green"]),
    ("Compound dry (S/M/H)", df["Compound"].isin(DRY)),
    ("Bukan lap 1 (standing start)", df["LapNumber"] > 1),
    ("TyreLife tersedia", df["TyreLife"].notna()),
    ("Bukan outlier pace", ~df["is_outlier"]),
]
m = pd.Series(True, index=df.index)
rows = []
for name, cond in steps:
    cond = cond.fillna(False).astype(bool)
    m &= cond
    rows.append((name, int(m.sum()), int((~cond).sum())))

funnel = pd.DataFrame(rows, columns=["Filter", "Sisa lap (kumulatif)", "Dibuang oleh filter ini saja"])
funnel["% sisa dari total"] = (funnel["Sisa lap (kumulatif)"] / len(df) * 100).round(1)
display(funnel)
print("Catatan: filter saling tumpang tindih (mis. semua in/out lap sudah IsAccurate=False), "
      "jadi kolom kumulatif bergantung urutan; gunakan kolom 'filter ini saja' untuk membaca dampak tiap filter.")

clean = df[m].copy()
clean.to_csv(CLEAN_PATH, index=False)
print(f"Clean racing lap: {len(clean):,} -> {CLEAN_PATH}")

# %%
# Sensitivitas IsAccurate: apakah lap yang lolos semua filter lain tetapi IsAccurate=False berbeda pace-nya?
others = (~df["is_missing_laptime"] & ~df["is_deleted"] & ~df["is_pit_lap"] & df["is_green"]
          & df["Compound"].isin(DRY) & (df["LapNumber"] > 1) & df["TyreLife"].notna() & ~df["is_outlier"])
acc = ~df["is_inaccurate"]
groups = {"IsAccurate = True": others & acc, "IsAccurate = False (lolos filter lain)": others & ~acc}
sens = pd.DataFrame({k: df.loc[v, "pace_delta"].agg(["count", "mean", "median", "std"]) for k, v in groups.items()}).T
display(sens.round(3))
print("Jika mean/median/std kedua kelompok mirip, IsAccurate tidak wajib dipakai; "
      "jika lap 'False' sangat menyimpang, pertahankan sebagai filter. Laporkan hasilnya.")

# %% [markdown]
# **Treatment yang dipakai (dokumentasikan alasannya di laporan)**
#
# | Kondisi | Treatment | Alasan |
# |---|---|---|
# | LapTime NaT / kosong | Dikeluarkan dari data latih | Target tidak ada; tidak diimputasi agar tidak menciptakan label palsu |
# | `IsAccurate = False` | Dikeluarkan (konservatif); dampaknya diuji di sel sensitivitas | Menurut dokumentasi FastF1, flag ini menandai sinkronisasi waktu mulai/akhir lap, **bukan** akurasi lap time. Pada data ini semua in/out lap bernilai False |
# | `Deleted` (lap dibatalkan steward) | **Tidak dapat dideteksi** | Kolom `Deleted`/`DeletedReason` kosong 100% karena race control messages tidak dimuat saat download; dicatat sebagai limitation |
# | Pit in-lap / out-lap | Dikeluarkan dari lap-time model; **pit loss diestimasi terpisah** | Waktunya didominasi pit lane, bukan degradasi ban |
# | SC / VSC / red flag / yellow | Dikeluarkan (hanya green flag) | Pace dikendalikan race control, bukan performa ban |
# | Compound INTERMEDIATE / WET | Dikeluarkan dari simulator dry | Karakter degradasi berbeda; disebut sebagai *limitation* |
# | Lap 1 | Dikeluarkan | Standing start + traffic |
# | Outlier pace (>115% atau <90% median race) | Dikeluarkan | Sisa insiden / traffic ekstrem; ambang bisa diuji di sensitivity analysis |

# %% [markdown]
# ### A.3 Visualisasi
# `pace_delta` = lap time dikurangi median race, agar lap dari sirkuit berbeda bisa dibandingkan.

# %%
# Viz 1: TyreLife vs pace_delta per compound
lo, hi = clean["pace_delta"].quantile([0.01, 0.99])
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
for ax, comp in zip(axes, DRY):
    d = clean[clean["Compound"] == comp]
    if d.empty:
        ax.set_title(f"{comp} (tidak ada data)")
        continue
    s = d.sample(min(len(d), 20000), random_state=42)
    ax.scatter(s["TyreLife"], s["pace_delta"], s=3, alpha=0.15, color=COMPOUND_COLORS[comp])
    mean = d.groupby("TyreLife")["pace_delta"].agg(["mean", "count"])
    mean = mean[mean["count"] >= 30]
    ax.plot(mean.index, mean["mean"], color="black", lw=2, label="rata-rata per TyreLife")
    ax.set_title(comp)
    ax.set_xlabel("TyreLife (lap)")
    ax.set_ylim(lo, hi)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Lap time - median race (detik)")
axes[0].legend()
fig.suptitle("Viz 1. TyreLife vs Lap Time (clean racing laps)")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/viz1_tyrelife_vs_laptime.png", dpi=150)
plt.show()

# %%
# Viz 2: Distribusi lap time per compound
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
data = [clean.loc[clean["Compound"] == c, "pace_delta"].dropna() for c in DRY]
axes[0].boxplot(data, showfliers=False)
axes[0].set_xticks(range(1, len(DRY) + 1))
axes[0].set_xticklabels(DRY)
axes[0].set_ylabel("Lap time - median race (detik)")
axes[0].set_title("Boxplot per compound")
for c, d in zip(DRY, data):
    axes[1].hist(d, bins=60, alpha=0.5, color=COMPOUND_COLORS[c], label=c, density=True)
axes[1].set_xlabel("Lap time - median race (detik)")
axes[1].set_title("Distribusi (density)")
axes[1].legend()
fig.suptitle("Viz 2. Distribusi Lap Time per Compound")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/viz2_laptime_by_compound.png", dpi=150)
plt.show()
display(clean.groupby("Compound")["pace_delta"].describe().round(3))

# %%
# Viz 3: Timeline stint / pit strategy (>= 2 pembalap dalam 1 race)
RACE_SEASON = int(df["Season"].max())
RACE_NAME = None      # None = race pertama pada musim tsb; atau isi mis. "Italian Grand Prix"
DRIVERS = None        # None = 2 finisher terbaik; atau isi mis. ["VER", "NOR"]

race_df = df[df["Season"] == RACE_SEASON]
if RACE_NAME is None:
    RACE_NAME = race_df["RaceName"].iloc[0]
race_df = race_df[race_df["RaceName"] == RACE_NAME]
if DRIVERS is None:
    last = race_df.sort_values("LapNumber").groupby("Driver").tail(1).sort_values("Position")
    DRIVERS = last["Driver"].head(2).tolist()


def stint_compound(s):
    s = s.dropna()
    return s.mode().iloc[0] if len(s) else "UNKNOWN"


fig, ax = plt.subplots(figsize=(12, 1.6 + 0.9 * len(DRIVERS)))
for i, drv in enumerate(DRIVERS):
    d = race_df[race_df["Driver"] == drv]
    for stint, g in d.groupby("Stint"):
        comp = stint_compound(g["Compound"])
        start, end = g["LapNumber"].min(), g["LapNumber"].max()
        ax.barh(i, end - start + 1, left=start - 1, color=COMPOUND_COLORS.get(comp, "#cccccc"),
                edgecolor="black")
        ax.text((start + end) / 2 - 0.5, i, comp[0], ha="center", va="center", fontsize=9, weight="bold")
ax.set_yticks(range(len(DRIVERS)))
ax.set_yticklabels(DRIVERS)
ax.invert_yaxis()
ax.set_xlabel("Lap")
handles = [plt.Rectangle((0, 0), 1, 1, color=COMPOUND_COLORS[c], ec="black") for c in COMPOUND_COLORS]
ax.legend(handles, list(COMPOUND_COLORS), ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.25))
ax.set_title(f"Viz 3. Timeline stint: {RACE_SEASON} {RACE_NAME}")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/viz3_stint_timeline.png", dpi=150)
plt.show()

stints = (race_df[race_df["Driver"].isin(DRIVERS)].groupby(["Driver", "Stint"])
          .agg(lap_awal=("LapNumber", "min"), lap_akhir=("LapNumber", "max"),
               compound=("Compound", stint_compound)))
display(stints)

# %% [markdown]
# ### A.4 Confounder: apakah degradasi ban bisa dibaca langsung?

# %%
# Demonstrasi confounder bahan bakar: bandingkan slope TyreLife dengan dan tanpa kontrol 'laps_remaining'.
# laps_remaining tinggi = tangki lebih berat = lebih lambat, sedangkan TyreLife naik seiring lap berjalan.
def ols(X, y):
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


rows = []
for comp in DRY:
    d = clean[clean["Compound"] == comp].dropna(subset=["pace_delta", "TyreLife", "laps_remaining"])
    if len(d) < 100:
        continue
    naive = ols(d[["TyreLife"]].to_numpy(), d["pace_delta"].to_numpy())
    ctrl = ols(d[["TyreLife", "laps_remaining"]].to_numpy(), d["pace_delta"].to_numpy())
    rows.append({"Compound": comp, "n": len(d),
                 "slope TyreLife (naif) s/lap": round(naive[1], 4),
                 "slope TyreLife (kontrol bbm) s/lap": round(ctrl[1], 4),
                 "koef laps_remaining s/lap": round(ctrl[2], 4)})
display(pd.DataFrame(rows))

# %% [markdown]
# **Potensi confounder (minimal 3)**
#
# | Confounder | Efek pada pembacaan degradasi | Mitigasi di pemodelan |
# |---|---|---|
# | **Fuel load menurun** sepanjang race | Mobil makin ringan sehingga lap time membaik, menutupi (menyamarkan) degradasi ban | Sertakan `laps_remaining` / `LapNumber` sebagai fitur |
# | **Traffic / dirty air** | Lap lambat karena di belakang mobil lain, bukan karena ban | Fitur posisi / gap (jika tersedia), buang outlier |
# | **Safety car / VSC / red flag** | Pace turun drastis; ban "istirahat" dan pit stop murah | Hanya pakai lap green flag; catat sebagai asumsi (SC tidak dimodelkan) |
# | **Cuaca & track evolution** | Suhu aspal dan grip berubah sepanjang race dan antar sesi | Fitur `TrackTemp`, `AirTemp`, `Rainfall`; normalisasi `pace_delta` per race |
# | **Perbedaan performa mobil / pembalap** | Team cepat terlihat "degradasi lebih kecil" | Fitur `Team` (dengan pemetaan nama antar musim) atau normalisasi per pembalap |
# | **Seleksi strategi** | Ban tua hanya muncul pada stint yang memang berjalan bagus (survivorship) | Perlu dibahas di limitation |
#
# **Interpretasi (isi setelah melihat grafik dan tabel di atas)**
# 1. Apakah degradasi ban terlihat pada Viz 1? Untuk compound mana?
# 2. Bagaimana slope berubah setelah kontrol bahan bakar?
# 3. Faktor apa yang paling mengganggu interpretasi pada data kamu?