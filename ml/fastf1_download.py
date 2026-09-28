"""
fastf1_download.py
=======================
Download data lap sesi RACE Formula 1 musim 2021-2025 memakai FastF1.


Cara pakai:
    pip install fastf1 pandas numpy
    python fastf1_download.py
Boleh dihentikan (Ctrl+C) dan dijalankan ulang kapan saja
"""

import os
import time
from datetime import datetime

import pandas as pd
import fastf1

try:
    from fastf1.exceptions import RateLimitExceededError
except ImportError:  # FastF1 versi lama
    from fastf1 import RateLimitExceededError

# =========================================================
# KONFIGURASI
# =========================================================
CACHE_DIR = "./fastf1_cache"
OUT_DIR = "./data"
RAW_DIR = os.path.join(OUT_DIR, "raw")
LOG_PATH = os.path.join(OUT_DIR, "download_log.csv")
SEASONS = [2021, 2022, 2023, 2024, 2025]
SESSION_TYPE = "R"        # R = Race (sprint tidak diambil)
WAIT_MINUTES = 10         # jeda saat kena rate limit
MAX_RETRY = 10            # maksimal percobaan ulang per race

WEATHER_COLS = ["AirTemp", "TrackTemp", "Humidity", "Pressure",
                "Rainfall", "WindSpeed", "WindDirection"]

KEEP_COLS = [
    "Season", "Round", "RaceName", "EventDate", "RaceLaps",
    "Driver", "DriverNumber", "Team",
    "LapNumber", "Stint", "Compound", "TyreLife", "FreshTyre",
    "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
    "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
    "Position", "TrackStatus", "IsAccurate",
    "PitInTime", "PitOutTime", "Deleted", "DeletedReason",
    "LapStartTime", "Time", "IsPersonalBest",
] + WEATHER_COLS

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)
fastf1.set_log_level("WARNING")


# =========================================================
# UTILITAS
# =========================================================
def is_rate_limit(e: Exception) -> bool:
    msg = str(e).lower()
    return (
        isinstance(e, RateLimitExceededError)
        or "rate limit" in msg
        or "calls/h" in msg
        or "429" in msg
        or "failed to load any schedule data" in msg  # FastF1 lama menyembunyikan rate limit
    )


def with_retry(func, *args):
    """Jalankan func; kalau kena rate limit, tunggu lalu ulangi."""
    for attempt in range(1, MAX_RETRY + 1):
        try:
            return func(*args)
        except Exception as e:
            if is_rate_limit(e) and attempt < MAX_RETRY:
                print(f"  [RATE LIMIT] menunggu {WAIT_MINUTES} menit, "
                      f"lalu coba lagi ({attempt}/{MAX_RETRY})...")
                time.sleep(WAIT_MINUTES * 60)
                continue
            raise


def get_schedule(season: int) -> pd.DataFrame:
    return fastf1.get_event_schedule(season, include_testing=False)


def load_race(season: int, rnd: int, race_name: str, event_date: str) -> pd.DataFrame:
    session = fastf1.get_session(season, rnd, SESSION_TYPE)
    session.load(laps=True, telemetry=False, weather=True, messages=False)

    laps = session.laps.reset_index(drop=True).copy()
    if laps.empty:
        return laps

    # Cuaca per lap (data cuaca sudah di-load; tidak memicu request baru)
    try:
        w = session.laps.get_weather_data().reset_index(drop=True)
        if len(w) == len(laps):
            for c in WEATHER_COLS:
                if c in w.columns:
                    laps[c] = w[c].to_numpy()
    except Exception as e:
        print(f"    [INFO] data cuaca tidak tersedia: {e}")

    laps["Season"] = season
    laps["Round"] = rnd
    laps["RaceName"] = race_name
    laps["EventDate"] = event_date
    laps["RaceLaps"] = laps["LapNumber"].max()

    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        if col in laps.columns:
            laps[col + "Seconds"] = laps[col].dt.total_seconds()

    cols = [c for c in KEEP_COLS if c in laps.columns]
    cols += [c for c in laps.columns if c.endswith("Seconds") and c not in cols]
    return laps[cols]


def load_log() -> dict:
    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
        return {(int(r.Season), int(r.Round)): r._asdict() for r in df.itertuples(index=False)}
    return {}


def save_log(log: dict):
    pd.DataFrame(list(log.values())).sort_values(["Season", "Round"]).to_csv(LOG_PATH, index=False)


def merge_outputs():
    by_season = {}
    for fn in sorted(os.listdir(RAW_DIR)):
        if fn.endswith(".csv"):
            season = int(fn.split("_")[0])
            by_season.setdefault(season, []).append(pd.read_csv(os.path.join(RAW_DIR, fn)))

    all_df = []
    for season, frames in sorted(by_season.items()):
        sdf = pd.concat(frames, ignore_index=True)
        sdf.to_csv(os.path.join(OUT_DIR, f"f1_laps_{season}.csv"), index=False)
        all_df.append(sdf)
        print(f"[MERGE] {season}: {sdf['Round'].nunique()} race, {len(sdf)} lap")

    if all_df:
        combined = pd.concat(all_df, ignore_index=True)
        path = os.path.join(OUT_DIR, "f1_laps_all_2021_2025.csv")
        combined.to_csv(path, index=False)
        print(f"[MERGE] Total: {combined.groupby(['Season', 'Round']).ngroups} race, "
              f"{len(combined)} lap -> {path}")


# =========================================================
# MAIN
# =========================================================
def main():
    log = load_log()

    for season in SEASONS:
        try:
            schedule = with_retry(get_schedule, season)
        except Exception as e:
            print(f"[ERROR] Kalender {season} gagal diambil: {e}")
            continue

        for _, ev in schedule.iterrows():
            rnd = int(ev["RoundNumber"])
            race_name = str(ev["EventName"])
            event_date = str(pd.Timestamp(ev["EventDate"]).date())
            raw_path = os.path.join(RAW_DIR, f"{season}_R{rnd:02d}.csv")

            if os.path.exists(raw_path) and os.path.getsize(raw_path) > 0:
                if (season, rnd) not in log:
                    n = len(pd.read_csv(raw_path, usecols=["Season"]))
                    log[(season, rnd)] = dict(Season=season, Round=rnd, RaceName=race_name,
                                              Status="OK", Laps=n, Message="sudah ada",
                                              Timestamp=datetime.now().isoformat(timespec="seconds"))
                continue

            print(f"[INFO] {season} R{rnd:02d} {race_name}")
            status, n_laps, msg = "OK", 0, ""
            try:
                df = with_retry(load_race, season, rnd, race_name, event_date)
                if df is None or df.empty:
                    status, msg = "EMPTY", "tidak ada data lap"
                else:
                    df.to_csv(raw_path, index=False)
                    n_laps = len(df)
            except Exception as e:
                status, msg = "FAILED", repr(e)[:250]
                print(f"  [GAGAL] {msg}")

            log[(season, rnd)] = dict(Season=season, Round=rnd, RaceName=race_name,
                                      Status=status, Laps=n_laps, Message=msg,
                                      Timestamp=datetime.now().isoformat(timespec="seconds"))
            save_log(log)

    save_log(log)
    merge_outputs()

    log_df = pd.DataFrame(list(log.values()))
    print("\n=== RINGKASAN STATUS DOWNLOAD ===")
    print(log_df.groupby(["Season", "Status"]).size().unstack(fill_value=0))
    bad = log_df[log_df["Status"] != "OK"]
    if len(bad):
        print("\nRace yang belum berhasil (jalankan ulang script untuk mencoba lagi):")
        print(bad[["Season", "Round", "RaceName", "Status", "Message"]].to_string(index=False))


if __name__ == "__main__":
    main()