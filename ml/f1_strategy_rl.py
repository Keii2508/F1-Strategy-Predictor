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
# ## Bagian B. Model Lap Time (Simulator Transisi)
#
# Tujuan: `f(state_t, action_t) -> predicted LapTime_(t+1)`, dilatih dari `clean` (hasil Bagian A).
#
# **Fitur yang dipakai (aman / diketahui sebelum lap dijalani):**
# `Circuit`, `TeamLineage` (nama team dipetakan lintas rebranding), `Compound`, `TyreLife`,
# `Stint`, `LapNumber`, `LapsRemaining` (proksi bahan bakar), `RollingPace3` (rata-rata 3 lap
# bersih sebelumnya, dalam race & driver yang sama).
#
# **Fitur yang SENGAJA tidak dipakai (potensi leakage):**
# `pace_delta` versi EDA (memakai median seluruh race — informasi masa depan), `Position` setelah
# lap (dipengaruhi hasil lap itu sendiri), `Sector*Time` dari lap yang sedang diprediksi.
#
# **Asumsi yang harus disebut di laporan:** `TrackTemp`/`AirTemp` memakai nilai terukur pada lap
# tersebut (bukan forecast) — sedikit optimis, tapi lazim di studi motorsport dan hampir tidak
# berubah antar lap yang berdekatan.

# %%
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import joblib

MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

# %% [markdown]
# ### B.1 Pemetaan nama team lintas musim (rebranding)

# %%
# FastF1 memberi nama team sesuai musim; beberapa entitas berganti nama.
# Pemetaan berikut mengelompokkan nama ke 'lineage' agar konsisten sebagai fitur.
# PERIKSA/LENGKAPI sesuai nama persis yang muncul di datamu (lihat daftar di bawah).
TEAM_LINEAGE = {
    "Alfa Romeo Racing": "Sauber_lineage", "Alfa Romeo": "Sauber_lineage",
    "Kick Sauber": "Sauber_lineage", "Sauber": "Sauber_lineage",
    "AlphaTauri": "RB_lineage", "RB": "RB_lineage", "Racing Bulls": "RB_lineage",
    "Racing Point": "Aston_lineage", "Aston Martin": "Aston_lineage",
}


def map_team(s: pd.Series) -> pd.Series:
    return s.map(lambda t: TEAM_LINEAGE.get(t, t))  # default: pakai nama asli bila tidak ada di mapping


print("Nama Team per musim (cek: apakah semua sudah tercakup mapping di atas?)")
display(df.groupby("Season")["Team"].apply(lambda s: sorted(s.dropna().unique())).to_frame("Team"))

# %% [markdown]
# ### B.2 Feature engineering (dari `clean`, hasil Bagian A)

# %%
feat = clean.sort_values(["Season", "RaceName", "Driver", "LapNumber"]).copy()
feat["TeamLineage"] = map_team(feat["Team"])
feat["Circuit"] = feat["RaceName"]              # 1 circuit = 1 race dalam dataset ini
feat["LapsRemaining"] = feat["RaceLaps"] - feat["LapNumber"]

# Rolling pace: rata-rata LapTimeSeconds dari lap-lap SEBELUMNYA (shift 1) pada
# race & driver yang sama. shift(1) + rolling mencegah lap saat ini "melihat" dirinya sendiri.
g = feat.groupby(["Season", "RaceName", "Driver"])["LapTimeSeconds"]
feat["RollingPace3"] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
feat["RollingPace3"] = feat["RollingPace3"].fillna(feat["LapTimeSeconds"].median())  # awal stint

FEATURES_NUM = ["TyreLife", "Stint", "LapNumber", "LapsRemaining", "RollingPace3"]
if "TrackTemp" in feat.columns:
    FEATURES_NUM += ["TrackTemp", "AirTemp"]
FEATURES_CAT = ["Compound", "Circuit", "TeamLineage"]
TARGET = "LapTimeSeconds"

model_df = feat.dropna(subset=FEATURES_NUM + FEATURES_CAT + [TARGET]).copy()
print(f"Baris siap dimodelkan: {len(model_df):,} dari {len(feat):,} clean lap")
model_df[["Season", "RaceName", "Driver"] + FEATURES_NUM + FEATURES_CAT + [TARGET]].head()

# %% [markdown]
# ### B.3 Split temporal (bukan random split!)

# %%
TRAIN_SEASONS = [2021, 2022, 2023]
VAL_SEASONS = [2024]
TEST_SEASONS = [2025]

train = model_df[model_df["Season"].isin(TRAIN_SEASONS)]
val = model_df[model_df["Season"].isin(VAL_SEASONS)]
test = model_df[model_df["Season"].isin(TEST_SEASONS)]

print(f"Train {TRAIN_SEASONS}: {len(train):,} lap, {train.groupby(['Season','RaceName']).ngroups} race")
print(f"Val   {VAL_SEASONS}: {len(val):,} lap, {val.groupby(['Season','RaceName']).ngroups} race")
print(f"Test  {TEST_SEASONS}: {len(test):,} lap, {test.groupby(['Season','RaceName']).ngroups} race")

# Sanity check wajib: pastikan tidak ada race yang sama muncul di lebih dari satu split
overlap = (set(zip(train.Season, train.RaceName)) & set(zip(val.Season, val.RaceName))
           | set(zip(train.Season, train.RaceName)) & set(zip(test.Season, test.RaceName)))
assert not overlap, f"BOCOR! Race berikut ada di >1 split: {overlap}"
print("OK: tidak ada race yang bocor antar split.")

# %% [markdown]
# ### B.4 Baseline: Linear Regression

# %%
def to_xy(d):
    return d[FEATURES_NUM + FEATURES_CAT], d[TARGET].to_numpy()


X_train, y_train = to_xy(train)
X_val, y_val = to_xy(val)
X_test, y_test = to_xy(test)

pre = ColumnTransformer([
    ("num", "passthrough", FEATURES_NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
])
lin_model = Pipeline([("pre", pre), ("lr", LinearRegression())])
lin_model.fit(X_train, y_train)


def report(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    print(f"{name:35s} MAE={mae:6.3f}s  RMSE={rmse:6.3f}s")
    return mae, rmse


print("--- Linear Regression (baseline) ---")
report("Train", y_train, lin_model.predict(X_train))
report("Val (2024)", y_val, lin_model.predict(X_val))
lr_test_mae, lr_test_rmse = report("Test (2025, unseen)", y_test, lin_model.predict(X_test))

joblib.dump(lin_model, f"{MODEL_DIR}/lap_time_linear.joblib")

# %% [markdown]
# ### B.5 Model utama: LightGBM

# %%
cat_idx = [FEATURES_NUM.__len__() + i for i in range(len(FEATURES_CAT))]  # tidak dipakai; LGBM di bawah pakai kolom category

lgb_train_df = train[FEATURES_NUM + FEATURES_CAT].copy()
lgb_val_df = val[FEATURES_NUM + FEATURES_CAT].copy()
lgb_test_df = test[FEATURES_NUM + FEATURES_CAT].copy()
for c in FEATURES_CAT:
    cats = pd.Categorical(train[c]).categories
    for d in (lgb_train_df, lgb_val_df, lgb_test_df):
        d[c] = pd.Categorical(d[c], categories=cats)

gbm = lgb.LGBMRegressor(
    n_estimators=1000, learning_rate=0.03, num_leaves=31,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=-1,
)
gbm.fit(
    lgb_train_df, y_train,
    eval_set=[(lgb_val_df, y_val)],
    eval_metric="mae",
    categorical_feature=FEATURES_CAT,
    callbacks=[lgb.early_stopping(50, verbose=False)],
)

print(f"--- LightGBM (best_iteration={gbm.best_iteration_}) ---")
report("Train", y_train, gbm.predict(lgb_train_df))
report("Val (2024)", y_val, gbm.predict(lgb_val_df))
gbm_test_mae, gbm_test_rmse = report("Test (2025, unseen)", y_test, gbm.predict(lgb_test_df))

joblib.dump(gbm, f"{MODEL_DIR}/lap_time_lgbm.joblib")

# %% [markdown]
# ### B.6 Ringkasan perbandingan & feature importance

# %%
summary = pd.DataFrame([
    {"Model": "Linear Regression", "MAE_test": lr_test_mae, "RMSE_test": lr_test_rmse},
    {"Model": "LightGBM", "MAE_test": gbm_test_mae, "RMSE_test": gbm_test_rmse},
]).round(4)
display(summary)
summary.to_csv(f"{MODEL_DIR}/comparison.csv", index=False)

imp = pd.Series(gbm.feature_importances_, index=FEATURES_NUM + FEATURES_CAT).sort_values()
fig, ax = plt.subplots(figsize=(7, 5))
imp.plot.barh(ax=ax, color="#1f6fd0")
ax.set_title("LightGBM feature importance")
ax.set_xlabel("importance (split count)")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/viz4_feature_importance.png", dpi=150)
plt.show()

# %% [markdown]
# ## Bagian C. RL Environment
#
# CSV statis bukan environment RL. Di sini transisi antar-lap disimulasikan oleh model
# LightGBM dari Bagian B (`f(state_t, action_t) -> predicted LapTime_(t+1)`), bukan diputar
# ulang dari data historis. Ini yang membedakan simulator model-based dari sekadar "replay CSV".
#
# | Elemen | Definisi |
# |---|---|
# | **State** | `LapNumber`, `LapsRemaining`, `Compound`, `TyreLife`, `Stint`, `RollingPace3` (pace 3 lap terakhir **hasil prediksi simulator**, bukan data asli) |
# | **Action** | 0 = Stay Out, 1 = Pit→Soft, 2 = Pit→Medium, 3 = Pit→Hard |
# | **Transisi** | Stay Out: `TyreLife += 1`. Pit: compound berganti, `TyreLife` reset ke 1, dan **pit-loss** (diestimasi per sirkuit dari data, bukan angka tetap) ditambahkan ke lap time. Lap time diprediksi oleh model Bagian B. |
# | **Reward** | `-(predicted_lap_time)` — pit-loss sudah termasuk di dalam predicted_lap_time saat aksi = pit |
# | **Episode** | 1 race, dari lap awal sampai `RaceLaps` |
#
# **Batasan penting yang harus disebut di laporan:** model Bagian B dilatih pada rentang `TyreLife`
# yang ada di data training. Kalau agent nanti memilih Stay Out sampai `TyreLife` jauh melampaui
# rentang itu, prediksi model menjadi ekstrapolasi yang tidak reliabel — environment ini **tidak**
# mencegahnya secara otomatis (lihat sel C.4 untuk pemeriksaan rentang training).

# %% [markdown]
# ### C.1 Estimasi pit-loss per sirkuit (dari data, bukan angka tetap)

# %%
import gymnasium as gym
from gymnasium import spaces

# Pit-loss = kelebihan waktu in-lap + out-lap dibanding 2x pace normal sirkuit itu.
# Dihitung dari df (SEBELUM filter is_pit_lap dibuang), memakai HANYA data training
# (2021-2023) supaya konsisten dengan prinsip 'tidak mengintip masa depan'.
train_seasons_for_env = TRAIN_SEASONS + VAL_SEASONS  # circuit baseline boleh pakai train+val

pit_rows = df[df["Season"].isin(train_seasons_for_env) & (df["is_pit_lap"])]
normal_pace = (clean[clean["Season"].isin(train_seasons_for_env)]
               .groupby("RaceName")["LapTimeSeconds"].median())

pit_loss_calc = (pit_rows.groupby(["RaceName", "Driver", "Season"])["LapTimeSeconds"]
                 .sum().rename("pit_laps_sum").reset_index())
pit_loss_calc["n_pit_laps"] = pit_rows.groupby(["RaceName", "Driver", "Season"]).size().values
pit_loss_calc = pit_loss_calc[pit_loss_calc["n_pit_laps"] == 2]  # in-lap + out-lap saja
pit_loss_calc["normal_2laps"] = pit_loss_calc["RaceName"].map(normal_pace) * 2
pit_loss_calc["pit_loss"] = pit_loss_calc["pit_laps_sum"] - pit_loss_calc["normal_2laps"]

PIT_LOSS_BY_CIRCUIT = (pit_loss_calc.groupby("RaceName")["pit_loss"]
                       .median().clip(lower=5).to_dict())  # clip: jaga2 dari nilai negatif/data aneh
DEFAULT_PIT_LOSS = float(np.median(list(PIT_LOSS_BY_CIRCUIT.values()))) if PIT_LOSS_BY_CIRCUIT else 22.0

print(f"Pit-loss median lintas sirkuit: {DEFAULT_PIT_LOSS:.1f} detik (dipakai sbg fallback)")
display(pd.Series(PIT_LOSS_BY_CIRCUIT, name="pit_loss_detik").sort_values().to_frame())
print("\nCatatan: sirkuit dengan sampel pit sedikit bisa memberi estimasi kurang stabil; "
      "jadikan bahan sensitivity analysis (ganti angka ini +/-20% lalu lihat perubahan strategi).")

# %% [markdown]
# ### C.2 Rentang training (untuk mendeteksi ekstrapolasi)

# %%
TRAINING_TYRELIFE_MAX = {c: int(train.loc[train["Compound"] == c, "TyreLife"].quantile(0.99))
                          for c in DRY}
print("Batas TyreLife (persentil-99 data training) per compound — dipakai sbg peringatan ekstrapolasi:")
print(TRAINING_TYRELIFE_MAX)

# %% [markdown]
# ### C.3 Environment Gymnasium

# %%
COMPOUND_LIST = DRY  # ["SOFT", "MEDIUM", "HARD"]
ACTION_NAMES = ["Stay Out", "Pit -> Soft", "Pit -> Medium", "Pit -> Hard"]


class F1StrategyEnv(gym.Env):
    """Environment 1 race. Transisi disimulasikan oleh model Bagian B (LightGBM).
    reward = -(predicted lap time), dijumlah 1 episode = -(total simulated race time).
    """
    metadata = {"render_modes": []}

    def __init__(self, model, reference_df, pit_loss_by_circuit, default_pit_loss,
                 tyrelife_max, circuit=None, team_lineage=None, race_laps=None, seed=None):
        super().__init__()
        self.model = model
        self.ref = reference_df  # dipakai untuk contoh sirkuit/tim/race_laps/suhu default jika tidak diisi manual
        self.pit_loss_by_circuit = pit_loss_by_circuit
        self.default_pit_loss = default_pit_loss
        self.tyrelife_max = tyrelife_max
        self._fixed_circuit = circuit
        self._fixed_team = team_lineage
        self._fixed_race_laps = race_laps
        self._rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(4)
        # Observasi: [LapNumber, LapsRemaining, TyreLife, Stint, RollingPace3,
        #             compound_onehot(3)]  -> semua dinormalisasi sederhana di _get_obs
        self.observation_space = spaces.Box(low=-10, high=10, shape=(8,), dtype=np.float32)

    def _sample_episode_context(self):
        if self._fixed_circuit is not None:
            circuit = self._fixed_circuit
        else:
            circuit = self._rng.choice(self.ref["Circuit"].unique())
        sub = self.ref[self.ref["Circuit"] == circuit]
        team = self._fixed_team or self._rng.choice(sub["TeamLineage"].unique())
        race_laps = self._fixed_race_laps or int(sub["LapsRemaining"].max() + sub["LapNumber"].min())
        track_temp = float(sub["TrackTemp"].mean()) if "TrackTemp" in sub.columns else 35.0
        air_temp = float(sub["AirTemp"].mean()) if "AirTemp" in sub.columns else 25.0
        start_compound = str(sub.sort_values("LapNumber")["Compound"].iloc[0])
        return dict(circuit=circuit, team=team, race_laps=race_laps,
                    track_temp=track_temp, air_temp=air_temp, start_compound=start_compound)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        ctx = self._sample_episode_context()
        self.circuit = ctx["circuit"]
        self.team = ctx["team"]
        self.race_laps = ctx["race_laps"]
        self.track_temp = ctx["track_temp"]
        self.air_temp = ctx["air_temp"]

        self.lap_number = 2          # lap 1 (standing start) tidak dimodelkan (lihat Bagian A)
        self.compound = ctx["start_compound"]
        self.tyre_life = 1
        self.stint = 1
        self.pace_history = []       # untuk RollingPace3, isi = lap time HASIL PREDIKSI sebelumnya
        self.total_time = 0.0
        self.log = []                # riwayat aksi utk dianalisis/dibandingkan
        return self._get_obs(), self._get_info()

    def _rolling_pace(self):
        if not self.pace_history:
            return float(self.ref["RollingPace3"].median())
        return float(np.mean(self.pace_history[-3:]))

    def _get_obs(self):
        onehot = [1.0 if self.compound == c else 0.0 for c in COMPOUND_LIST]
        return np.array([
            self.lap_number / self.race_laps,
            (self.race_laps - self.lap_number) / self.race_laps,
            self.tyre_life / 40.0,
            self.stint / 4.0,
            self._rolling_pace() / 100.0,
            *onehot,
        ], dtype=np.float32)

    def _get_info(self):
        return dict(circuit=self.circuit, lap_number=self.lap_number, compound=self.compound,
                    tyre_life=self.tyre_life, total_time=self.total_time,
                    tyrelife_exceeded=self.tyre_life > self.tyrelife_max.get(self.compound, 999))

    def step(self, action: int):
        assert self.action_space.contains(action)
        pit_loss = 0.0
        if action == 0:
            self.tyre_life += 1
        else:
            self.compound = COMPOUND_LIST[action - 1]
            self.tyre_life = 1
            self.stint += 1
            pit_loss = self.pit_loss_by_circuit.get(self.circuit, self.default_pit_loss)

        x = pd.DataFrame([{
            "TyreLife": self.tyre_life, "Stint": self.stint, "LapNumber": self.lap_number,
            "LapsRemaining": self.race_laps - self.lap_number, "RollingPace3": self._rolling_pace(),
            "TrackTemp": self.track_temp, "AirTemp": self.air_temp,
            "Compound": self.compound, "Circuit": self.circuit, "TeamLineage": self.team,
        }])
        for c in FEATURES_CAT:
            x[c] = pd.Categorical(x[c], categories=pd.Categorical(self.ref[c]).categories)

        pred_pace = float(self.model.predict(x[FEATURES_NUM + FEATURES_CAT])[0])
        lap_time = pred_pace + pit_loss
        self.pace_history.append(pred_pace)   # pace 'murni' (tanpa pit-loss) utk RollingPace3
        self.total_time += lap_time
        reward = -lap_time

        self.log.append(dict(lap=self.lap_number, action=ACTION_NAMES[action],
                              compound=self.compound, tyre_life=self.tyre_life,
                              lap_time=round(lap_time, 3)))

        self.lap_number += 1
        terminated = self.lap_number > self.race_laps
        truncated = False
        return self._get_obs(), reward, terminated, truncated, self._get_info()

# %% [markdown]
# ### C.4 Uji environment: `check_env` + rollout kebijakan acak

# %%
from gymnasium.utils.env_checker import check_env

env = F1StrategyEnv(
    model=gbm, reference_df=model_df, pit_loss_by_circuit=PIT_LOSS_BY_CIRCUIT,
    default_pit_loss=DEFAULT_PIT_LOSS, tyrelife_max=TRAINING_TYRELIFE_MAX,
    circuit="Azerbaijan Grand Prix", seed=42,
)
check_env(env.unwrapped if hasattr(env, "unwrapped") else env, warn=True, skip_render_check=True)
print("check_env: OK (tidak ada AssertionError di atas)")

obs, info = env.reset(seed=0)
done = False
n_pits = 0
while not done:
    # Kebijakan acak sederhana: 95% Stay Out, sisanya pit (hanya utk sanity check environment,
    # BUKAN baseline yang dilaporkan -> baseline sebenarnya ada di Bagian D)
    action = 0 if np.random.rand() < 0.95 else np.random.randint(1, 4)
    obs, reward, done, truncated, info = env.step(action)
    if action != 0:
        n_pits += 1

print(f"\nSirkuit  : {info['circuit']}")
print(f"Total lap: {env.race_laps} | Jumlah pit: {n_pits}")
print(f"Total simulated race time: {env.total_time:.1f} detik "
      f"({env.total_time/60:.1f} menit)")
print(f"Lap terakhir melebihi rentang training TyreLife? {info['tyrelife_exceeded']}")

log_df = pd.DataFrame(env.log)
display(log_df.head(10))
display(log_df[log_df['action'] != 'Stay Out'])

# %%
sorted(model_df['Circuit'].unique())

# %% [markdown]
# ## Bagian D. Agent & Baseline
#
# Tiga strategi dibandingkan pada **race test (2025) yang tidak pernah dilihat** model Bagian B
# maupun agent Bagian D, memakai simulator (environment) yang **sama** untuk ketiganya supaya adil:
#
# 1. **Historical** — urutan pit & compound yang benar-benar dipakai pembalap, "diputar ulang" lewat simulator.
# 2. **Rule-based** — pit saat `TyreLife` melewati threshold yang dihitung dari data training (bukan ditebak).
# 3. **RL (PPO)** — agent yang dilatih di environment Bagian C.
#
# **Peringatan metodologis (wajib dibaca sebelum menilai hasil):** ketiganya dievaluasi lewat
# prediksi model Bagian B, BUKAN waktu asli dari data. Artinya yang dibandingkan adalah
# "seberapa baik tiap strategi menurut simulator", bukan "siapa yang pasti menang di balapan
# sungguhan". Kalau RL terlihat menang telak, periksa dulu apakah itu karena memang strategi
# yang lebih baik, atau karena RL menemukan celah/bias di simulator (lihat sel D.5).

# %% [markdown]
# ### D.1 Baseline 1: Historical strategy (replay lewat simulator)

# %%
def extract_historical_actions(raw_df, season, race_name, driver):
    """Ambil urutan compound asli dari data mentah lap demi lap, lalu ubah jadi urutan
    action (0=Stay Out, 1/2/3=Pit ke Soft/Medium/Hard) yang bisa 'diputar ulang' di environment.
    Mengembalikan None kalau race/driver ini memakai ban basah (di luar cakupan simulator)."""
    d = raw_df[(raw_df["Season"] == season) & (raw_df["RaceName"] == race_name)
               & (raw_df["Driver"] == driver)].sort_values("LapNumber")
    d = d[d["LapNumber"] >= 2]
    if d.empty or not d["Compound"].isin(DRY).all():
        return None  # race basah / data compound tidak lengkap -> skip dari perbandingan ini

    actions, start_compound = [], str(d["Compound"].iloc[0])
    prev_compound = start_compound
    for _, row in d.iterrows():
        comp = str(row["Compound"])
        if comp != prev_compound:
            actions.append(COMPOUND_LIST.index(comp) + 1)  # pit ke compound baru
        else:
            actions.append(0)  # stay out
        prev_compound = comp
    return dict(start_compound=start_compound, actions=actions, race_laps=len(d) + 1,
                team=str(d["Team"].iloc[0]))


def run_forced_actions(circuit, team, race_laps, start_compound, actions, model, ref_df,
                        pit_loss_tbl, default_pit_loss, tyrelife_max, seed=0):
    """Jalankan environment dengan urutan action yang SUDAH ditentukan (bukan dari policy).
    Dipakai untuk replay historical maupun eksekusi rule-based/RL yang sudah memilih aksinya."""
    env = F1StrategyEnv(model=model, reference_df=ref_df, pit_loss_by_circuit=pit_loss_tbl,
                         default_pit_loss=default_pit_loss, tyrelife_max=tyrelife_max,
                         circuit=circuit, team_lineage=team, race_laps=race_laps, seed=seed)
    obs, info = env.reset(seed=seed)
    env.compound = start_compound   # override start compound agar sama persis dgn historis
    for a in actions:
        obs, reward, done, truncated, info = env.step(a)
        if done:
            break
    return env

# %% [markdown]
# ### D.2 Baseline 2: Rule-based (threshold dari data training)

# %%
# Threshold = persentil-75 panjang stint per compound, dihitung HANYA dari data training.
stint_len = (train.groupby(["RaceName", "Driver", "Stint", "Compound"])["TyreLife"]
             .max().reset_index())
RULE_THRESHOLD = stint_len.groupby("Compound")["TyreLife"].quantile(0.75).to_dict()
print("Threshold pit (persentil-75 panjang stint, data training):", RULE_THRESHOLD)


def rule_based_policy(race_laps, start_compound, threshold=RULE_THRESHOLD):
    """Strategi 2-stop sederhana: mulai di start_compound, pit ke MEDIUM saat threshold
    compound saat ini terlampaui, lalu pit ke HARD saat threshold MEDIUM terlampaui,
    bertahan di HARD sampai finish. Ini BUKAN strategi optimal -> memang hanya baseline."""
    actions = []
    compound, tyre_life = start_compound, 1
    order_after = {"SOFT": "MEDIUM", "MEDIUM": "HARD", "HARD": "HARD"}
    for lap in range(2, race_laps + 1):
        laps_left = race_laps - lap
        if tyre_life >= threshold.get(compound, 30) and compound != "HARD" and laps_left > 3:
            nxt = order_after[compound]
            actions.append(COMPOUND_LIST.index(nxt) + 1)
            compound, tyre_life = nxt, 1
        else:
            actions.append(0)
            tyre_life += 1
    return actions

# %% [markdown]
# ### D.3 Melatih agent RL (PPO)

# %%
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env as sb3_check_env
from stable_baselines3.common.monitor import Monitor

MODEL_RL_PATH = f"{MODEL_DIR}/ppo_strategy.zip"

train_env = Monitor(F1StrategyEnv(
    model=gbm, reference_df=model_df, pit_loss_by_circuit=PIT_LOSS_BY_CIRCUIT,
    default_pit_loss=DEFAULT_PIT_LOSS, tyrelife_max=TRAINING_TYRELIFE_MAX,
    seed=1,
))  # circuit TIDAK difix -> tiap episode sirkuit diacak dari model_df (generalisasi lintas sirkuit)
sb3_check_env(train_env.unwrapped)

TOTAL_TIMESTEPS = 20_000  # starter; naikkan (mis. 200_000+) untuk hasil yang lebih matang

ppo_agent = PPO("MlpPolicy", train_env, verbose=0, seed=1,
                 n_steps=512, batch_size=64, learning_rate=3e-4)
ppo_agent.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
ppo_agent.save(MODEL_RL_PATH)
print(f"Training selesai ({TOTAL_TIMESTEPS:,} timesteps). Model disimpan: {MODEL_RL_PATH}")
print("PERINGATAN: ini baru starter run. Naikkan TOTAL_TIMESTEPS dan evaluasi reward curve "
      "(sel D.3b) sebelum dipakai sbg hasil akhir laporan.")

# %% [markdown]
# #### D.3b Reward curve (wajib untuk laporan: cek stabilitas training)

# %%
log_path = train_env.get_episode_rewards()
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(log_path, alpha=0.4, label="per episode")
if len(log_path) >= 10:
    smoothed = pd.Series(log_path).rolling(10).mean()
    ax.plot(smoothed, color="black", lw=2, label="rolling mean (10 episode)")
ax.set_xlabel("Episode")
ax.set_ylabel("Total reward (= -total simulated race time)")
ax.set_title("Reward curve PPO")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/viz5_reward_curve.png", dpi=150)
plt.show()
print(f"Episode tercatat: {len(log_path)}")

# %% [markdown]
# ### D.4 Perbandingan di race test (2025, unseen)

# %%
def rl_policy_actions(circuit, team, race_laps, start_compound, agent, model, ref_df,
                       pit_loss_tbl, default_pit_loss, tyrelife_max, seed=0):
    env = F1StrategyEnv(model=model, reference_df=ref_df, pit_loss_by_circuit=pit_loss_tbl,
                         default_pit_loss=default_pit_loss, tyrelife_max=tyrelife_max,
                         circuit=circuit, team_lineage=team, race_laps=race_laps, seed=seed)
    obs, info = env.reset(seed=seed)
    env.compound = start_compound
    obs = env._get_obs()
    actions = []
    for _ in range(2, race_laps + 1):
        action, _ = agent.predict(obs, deterministic=True)
        actions.append(int(action))
        obs, reward, done, truncated, info = env.step(int(action))
        if done:
            break
    return actions


results = []
test_pairs = (test[["Season", "RaceName", "Driver"]].drop_duplicates().itertuples(index=False))
for season, race_name, driver in test_pairs:
    hist = extract_historical_actions(df, season, race_name, driver)
    if hist is None:
        continue  # race basah / data tidak lengkap -> dilewati dari perbandingan head-to-head

    common = dict(circuit=race_name, team=hist["team"], race_laps=hist["race_laps"],
                  model=gbm, ref_df=model_df, pit_loss_tbl=PIT_LOSS_BY_CIRCUIT,
                  default_pit_loss=DEFAULT_PIT_LOSS, tyrelife_max=TRAINING_TYRELIFE_MAX)

    env_hist = run_forced_actions(start_compound=hist["start_compound"],
                                   actions=hist["actions"], **common)
    rb_actions = rule_based_policy(hist["race_laps"], hist["start_compound"])
    env_rb = run_forced_actions(start_compound=hist["start_compound"],
                                 actions=rb_actions, **common)
    rl_actions = rl_policy_actions(start_compound=hist["start_compound"], agent=ppo_agent, **common)
    env_rl = run_forced_actions(start_compound=hist["start_compound"],
                                 actions=rl_actions, **common)

    for label, env_x, acts in [("Historical", env_hist, hist["actions"]),
                                ("Rule-based", env_rb, rb_actions),
                                ("RL (PPO)", env_rl, rl_actions)]:
        results.append(dict(
            Season=season, RaceName=race_name, Driver=driver, Strategy=label,
            total_time_s=round(env_x.total_time, 1),
            n_pits=sum(1 for a in acts if a != 0),
            compounds_used=len(set([hist['start_compound']] +
                                   [COMPOUND_LIST[a-1] for a in acts if a != 0])),
            tyrelife_exceeded=env_x._get_info()["tyrelife_exceeded"],
        ))

comparison_df = pd.DataFrame(results)
if comparison_df.empty:
    print("Tidak ada pasangan race/driver dry yang bisa dibandingkan di test set "
          "(mungkin semua race test basah, atau data compound tidak lengkap).")
else:
    display(comparison_df.sort_values(["RaceName", "Driver", "Strategy"]))
    comparison_df.to_csv(f"{MODEL_DIR}/strategy_comparison.csv", index=False)

    pivot = comparison_df.pivot_table(index=["RaceName", "Driver"], columns="Strategy",
                                       values="total_time_s")
    print("\nSelisih waktu vs Historical (negatif = lebih cepat dari historical):")
    for col in ["Rule-based", "RL (PPO)"]:
        if col in pivot.columns:
            pivot[f"{col} vs Historical"] = pivot[col] - pivot["Historical"]
    display(pivot.round(1))

# %% [markdown]
# ### D.5 Generalisasi & sanity check (apakah RL benar-benar pintar, atau mengeksploitasi simulator?)

# %%
if not comparison_df.empty:
    n_races_compared = comparison_df[["RaceName", "Driver"]].drop_duplicates().shape[0]
    print(f"Jumlah pasangan race-driver yang dibandingkan: {n_races_compared} "
          f"(syarat tugas: >= 2 race unseen)")

    rl_rows = comparison_df[comparison_df["Strategy"] == "RL (PPO)"]
    print(f"\nRL memilih TyreLife di luar rentang training pada "
          f"{rl_rows['tyrelife_exceeded'].sum()} dari {len(rl_rows)} race "
          f"-> kalau > 0, periksa apakah 'kemenangan' RL di race itu valid atau "
          f"hasil ekstrapolasi model yang tidak reliabel.")

    print(f"\nRata-rata jumlah pit -> Historical: {comparison_df[comparison_df.Strategy=='Historical'].n_pits.mean():.2f}, "
          f"Rule-based: {comparison_df[comparison_df.Strategy=='Rule-based'].n_pits.mean():.2f}, "
          f"RL: {comparison_df[comparison_df.Strategy=='RL (PPO)'].n_pits.mean():.2f}")
    print(f"Rata-rata jml compound berbeda dipakai -> RL: "
          f"{rl_rows['compounds_used'].mean():.2f} "
          f"(aturan FIA sungguhan mewajibkan >= 2 compound berbeda saat kering -> "
          f"environment ini BELUM memaksakan itu, lihat diskusi di Bagian C)")

# %% [markdown]
# **Pertanyaan kunci yang harus dijawab di laporan (isi setelah melihat hasil di atas)**
#
# > *Apakah agent benar-benar menemukan strategi yang lebih efisien pada race unseen, atau hanya
# > mengeksploitasi kelemahan simulator yang kalian buat?*
#
# Bukti yang perlu dicek sebelum menjawab:
# 1. Apakah RL konsisten lebih cepat dari historical & rule-based di **sebagian besar** race test,
#    atau cuma menang di 1-2 race (indikasi hasil kebetulan)?
# 2. Apakah kemenangan RL berbarengan dengan `tyrelife_exceeded = True`? Jika ya, kemenangan itu
#    kemungkinan besar **artefak ekstrapolasi model**, bukan strategi yang valid.
# 3. Apakah RL memakai jumlah pit yang masuk akal (1-3), atau mencoba pit berkali-kali karena
#    entah bagaimana itu "menguntungkan" di mata simulator (celah reward)?
# 4. `TOTAL_TIMESTEPS` di D.3 masih kecil (starter run). Sebelum menarik kesimpulan final, latih
#    lebih lama dan lihat apakah reward curve di D.3b benar-benar konvergen (mendatar), bukan
#    masih naik tajam saat training dihentikan.
#
# **Ablation/sensitivity yang disyaratkan tugas (pilih minimal 1, kerjakan di sel terpisah):**
# - Ubah `PIT_LOSS_BY_CIRCUIT` (mis. +/-20%) lalu latih ulang/evaluasi ulang — apakah strategi RL berubah drastis?
# - Ubah `RULE_THRESHOLD` rule-based, lihat apakah historical vs rule-based masih konsisten unggul/kalah.
# - Bandingkan hasil kalau constraint 'minimal 2 compound' DIPAKSAKAN (lihat diskusi Bagian C) vs tidak.