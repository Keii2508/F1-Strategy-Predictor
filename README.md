# F1 Strategy Predictor

Proyek pre-thesis: **Implementasi Pendekatan Supervised Learning untuk Prediksi Degradasi Ban dan Optimalisasi Strategi Balap Formula 1**.

Tugas mata kuliah: model lap time (supervised learning) sebagai simulator, lalu agent Reinforcement Learning yang memilih aksi *Stay Out / Pit Soft / Pit Medium / Pit Hard*, dievaluasi pada race yang tidak dipakai saat training.

> Hasil strategi adalah strategi optimal **di dalam simulator yang kami bangun**, bukan bukti bahwa strategi itu pasti menang di balapan nyata.

## Struktur repo

```
.
├── app/, routes/, resources/, ...   # Laravel (website)
└── ml/                              # data science
    ├── fastf1_download.py           # download data lap 2021-2025 (checkpoint per race)
    ├── audit_data.py                # audit kualitas data
    ├── f1_strategy_rl.ipynb         # notebook utama
    ├── requirements.txt
    └── data/                        # TIDAK di-commit (kecuali download_log.csv)
```

## Data provenance

- Sumber: data timing Formula 1 (F1 Live Timing API) yang diakses lewat library [FastF1](https://docs.fastf1.dev/) (unofficial, tidak berafiliasi dengan F1). Data historis/jadwal juga memakai Jolpica-F1.
- CSV dibuat oleh `ml/fastf1_download.py` (`fastf1.get_session(season, round, "R")`, `session.load()`, `session.laps`). Kolom `Season`, `Round`, `RaceName`, `RaceLaps`, dan kolom cuaca ditambahkan oleh script.
- Status unduhan tiap race: `ml/data/download_log.csv` (ikut di-commit sebagai bukti kelengkapan).
- Keterbatasan yang diketahui: kolom `Deleted`/`DeletedReason` kosong karena race control messages tidak dimuat.

## Setup bagian ML (Python)

```powershell
cd ml
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Mendapatkan data

Pilih salah satu:

- **A (cepat, disarankan):** unduh `f1_laps_all_2021_2025.csv` dari `[ISI LINK GOOGLE DRIVE / GITHUB RELEASE]`, lalu taruh di `ml/data/`.
- **B (dari nol):** `python fastf1_download.py`. Bisa memakan beberapa jam karena rate limit FastF1 (500 request/jam); script menunggu otomatis dan bisa dihentikan lalu dilanjutkan.

Setelah data ada: `python audit_data.py` untuk laporan kualitas data.

### Menjalankan notebook

Jalankan dari dalam folder `ml/` (path data bersifat relatif):

```powershell
cd ml
python -m jupyter notebook
```

## Setup bagian web (Laravel)

```powershell
composer install
copy .env.example .env
php artisan key:generate
# atur DB_* di .env (buat database di Laragon), lalu:
php artisan migrate
php artisan serve
```

## Workflow tim

1. `git pull` sebelum mulai bekerja.
2. Satu orang, satu branch: `git checkout -b nama/fitur` (contoh: `budi/lap-time-model`). Jangan commit langsung ke `main`.
3. Buka Pull Request; anggota lain me-review sebelum merge.
4. Hindari dua orang mengedit **notebook yang sama** bersamaan (file `.ipynb` sulit di-merge). Pisahkan notebook per bagian/orang.
5. Sebelum commit notebook, bersihkan output:
   `python -m jupyter nbconvert --clear-output --inplace nama_notebook.ipynb`
6. **Jangan pernah commit** `.env`, `ml/data/*.csv`, atau `ml/fastf1_cache/`.

## Pembagian tugas

| Bagian | Isi | PIC |
|---|---|---|
| A | Data audit & EDA | |
| B | Model lap time (Linear Regression, LightGBM) | |
| C | RL environment (Gymnasium) | |
| D | Agent, baseline, evaluasi | |
| Web | Laravel | |
