# D3 Final Notebook — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `deliverable_3/d3.ipynb` — a standalone, reproducible final notebook that integrates D1+D2 into one heat-risk-screening narrative and adds four new analyses (spatial features, cross-city transfer, bootstrapped uncertainty, learning curves).

**Architecture:** A Python assembly script `deliverable_3/build_d3.py` declares the full ordered cell list (markdown + code) and writes `d3.ipynb`; the notebook is then executed with `jupyter nbconvert --to notebook --execute --inplace`. This is the proven D1/D2 build pattern. The notebook is self-contained: own SEF-parsing + risk-metric + feature code, reads data from `../archive/`, caches parquet to `deliverable_3/cache/`. Most backbone code is ported verbatim from the working `deliverable_2/d2.ipynb`; the four new analyses are net-new.

**Tech Stack:** Python 3, pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter/nbconvert. Kernel name `cdap`. No new dependencies (all pinned in repo-root `requirements.txt`).

## Global Constraints

- Notebook reads the dataset ONLY from the shared repo-root `../archive/`. Do NOT duplicate the dataset into `deliverable_3/`. Do NOT add Colab/Kaggle/web-fetch paths.
- `station_id` = filename stem before `_AirTemp_`, Unicode-normalized to **NFC** in BOTH the parser and the metadata load (macOS filesystem is NFD; metadata TSVs are NFC). Missing sentinels: literal `NA` and `-999`/`-999.0`.
- `RANDOM_STATE = 42` everywhere a seed is accepted.
- No new external data. No synthesised feature values. Real land cover stays named as a limitation only.
- Cross-references in markdown written as "Section 3.1" — NEVER the `§` symbol.
- Every markdown section opens with an elaborate plain-language explanation aimed at a first-time reader. Rubric category names are NOT echoed in the notebook.
- Plots: every axis labelled with units; degrees Celsius written as "degC" in tabular/column contexts (not the `°C` glyph) to match D2.
- Author name in any author/footer text: **Kriti Shrestha** (and Jonas Juchelka). Not "Sunny Tamrakar".
- Leak-free: all fitted state (imputer medians, scaler stats, city centroids, kNN neighbours, kNN-neighbour p95) learned on TRAIN folds only.
- Verification after any notebook build: 0 execution errors, 0 occurrences of `§`, 0 occurrences of the literal placeholder text `fill after running`.

---

## File structure

- **Create:** `deliverable_3/build_d3.py` — assembly script; single source of truth for cell content. One Python list `CELLS` of `("md"|"code", source_string)`, a writer that emits `d3.ipynb`.
- **Create:** `deliverable_3/d3.ipynb` — generated artifact (executed in place).
- **Create:** `deliverable_3/cache/hourly_long.parquet` — generated cache (git-ignored).
- **Reference (read-only):** `deliverable_2/d2.ipynb` (source of ported backbone code), `deliverable_3/D3_design.md` (the approved spec).

The build script is the unit we edit. Each task appends/edits cells in `build_d3.py`, regenerates and executes `d3.ipynb`, and verifies. Interpretation prose that depends on computed numbers is written as a literal `fill after running` placeholder in early tasks and replaced with real numbers in Task 12 (after every number exists in executed output).

### Standard build-and-verify commands (used by every task)

Regenerate the notebook from the script:
```bash
cd deliverable_3 && python build_d3.py
```
Expected: prints `Wrote d3.ipynb (<N> cells)`.

Execute the notebook in place (kernel `cdap`):
```bash
cd deliverable_3 && jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=cdap --ExecutePreprocessor.timeout=600 d3.ipynb
```
Expected: `[NbConvertApp] Writing ... bytes to d3.ipynb` and exit code 0 (no traceback).

Quick integrity scan (run after execution):
```bash
cd deliverable_3 && python -c "
import json; nb=json.load(open('d3.ipynb')); errs=0; secs=0; fills=0
for c in nb['cells']:
    if c['cell_type']=='code':
        for o in c.get('outputs',[]):
            if o.get('output_type')=='error': errs+=1
    src=''.join(c['source']); secs+=src.count(chr(167)); fills+=src.lower().count('fill after running')
print(f'cells={len(nb[\"cells\"])} errors={errs} section-symbols={secs} fill-after-running={fills}')
"
```

---

## Task 1: Build script skeleton + Section 0 setup, executes clean

**Files:**
- Create: `deliverable_3/build_d3.py`
- Create (generated): `deliverable_3/d3.ipynb`

**Interfaces:**
- Produces: `build_d3.py` exposing module-level `CELLS` (list of `(kind, src)`), `md(s)`/`code(s)` helpers, and a `__main__` writer that emits `d3.ipynb` with `metadata.kernelspec.name = "cdap"`.

- [ ] **Step 1: Write `build_d3.py` skeleton with the writer and first two cells**

```python
"""Assemble deliverable_3/d3.ipynb from an ordered cell list. Run: python build_d3.py"""
import json
from pathlib import Path

CELLS = []
def md(s):   CELLS.append(("markdown", s.strip("\n")))
def code(s): CELLS.append(("code", s.strip("\n")))

# ---- Title -------------------------------------------------------------
md(r"""
# Heat-Risk Screening — Deliverable 3 (Final Integrated Notebook)

**Course:** Computational Data Analysis with Python (CDAP), TUM
**Track 1 — Heat-risk screening** · Kriti Shrestha & Jonas Juchelka

This notebook is the final, standalone synthesis of the project. It runs top to
bottom from a clean kernel and does not depend on the Deliverable 1 or 2
notebooks. It carries its own data-loading code and reads the shared dataset
from `../archive/`.

We study urban summer heat exposure across six European cities and build a
screening tool that flags the hottest measurement locations. The notebook walks
the full path from the question, through the data and its quality, the analysis
and its validation, to an honest read of uncertainty and a concrete
recommendation a city could act on.
""")

# ---- Section 0 ---------------------------------------------------------
md(r"""
## 0. Setup

We import the scientific Python stack and scikit-learn, fix a single random
seed so every result is reproducible, and set a consistent plotting style.
Nothing here is analysis yet; it just prepares the environment.
""")

code(r"""
from pathlib import Path
import re
import unicodedata
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import (
    StratifiedKFold, GroupKFold, cross_val_predict, cross_val_score,
    GridSearchCV, learning_curve,
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    precision_score, recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, ConfusionMatrixDisplay,
)

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams['figure.figsize'] = (10, 5)
warnings.filterwarnings('ignore', category=FutureWarning)

print('Libraries loaded.')
print(f'pandas {pd.__version__}, numpy {np.__version__}, sklearn loaded')
""")

# ---- writer ------------------------------------------------------------
def build():
    cells = []
    for kind, src in CELLS:
        lines = src.splitlines(keepends=True)
        cell = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cells.append(cell)
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python (CDAP)", "language": "python", "name": "cdap"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    Path("d3.ipynb").write_text(json.dumps(nb, indent=1))
    print(f"Wrote d3.ipynb ({len(cells)} cells)")

if __name__ == "__main__":
    build()
```

- [ ] **Step 2: Generate the notebook**

Run: `cd deliverable_3 && python build_d3.py`
Expected: `Wrote d3.ipynb (4 cells)`

- [ ] **Step 3: Execute the notebook**

Run the standard execute command (kernel `cdap`).
Expected: exit code 0; `Libraries loaded.` appears in output; no traceback.

- [ ] **Step 4: Integrity scan**

Run the standard integrity scan.
Expected: `cells=4 errors=0 section-symbols=0 fill-after-running=0`

- [ ] **Step 5: Commit**

```bash
git add deliverable_3/build_d3.py deliverable_3/d3.ipynb
git commit -m "feat(d3): build-script skeleton + setup cell"
```

---

## Task 2: Data loading + quality (Section 2), 375 stations verified

**Files:**
- Modify: `deliverable_3/build_d3.py` (append Section 1 markdown, Section 2 markdown + loader code)

**Interfaces:**
- Consumes: imports from Task 1.
- Produces: in-notebook DataFrame `hourly` with columns `city, station_id, datetime, ta`; parquet cache at `cache/hourly_long.parquet`.

- [ ] **Step 1: Append Section 1 (question) heading + Section 2 markdown + loader code to `build_d3.py`** (before the `def build()` writer)

Section 1 is prose-only here (full motivation text finalised in Task 12); add a short stub now so numbering is stable:

```python
md(r"""
## 1. Question and motivation

`fill after running` — elaborate motivation written in Task 12.
""")

md(r"""
## 2. The data

The dataset is **FAIRUrbTemp**, a curated collection of urban air-temperature
measurements from European cities. Each city has a set of fixed measurement
stations, and each station records hourly **air temperature** (`ta`) in degrees
Celsius. Air temperature is the only measured variable; everything else we use
is station context (where the station is, how high its sensor sits).

The files are stored in **SEF (Station Exchange Format)**: a 12-line header
with station identity and units, then a tab-separated table whose key column is
`Value` (the temperature) alongside `Year, Month, Day, Hour, Minute`. Missing
readings appear either as the literal text `NA` or as the sentinel `-999`; we
fold both into proper missing values so they never pollute a temperature
average. Our **unit of analysis is the station**: every row of the modeling
table we build later is one station.
""")

md(r"""
### 2.1 Load the hourly records

We parse every station file into one long table. Two details matter for
correctness. First, a station's id is the filename stem before `_AirTemp_`,
which keeps compound ids such as `AWEL_Blumenfeld` intact. Second, we normalise
ids to Unicode NFC so that stations with umlauts (for example Basel's
`Münsterplatz`) match the metadata files later; macOS stores the filenames in a
different normalisation (NFD) than the metadata tables (NFC), and without this
step those stations would silently fail to merge.
""")

code(r"""
ARCHIVE = Path('../archive')
HOURLY_DIR = ARCHIVE / 'data' / 'hourly'
META_DIR   = ARCHIVE / 'data' / 'metadata'
CACHE_DIR  = Path('cache'); CACHE_DIR.mkdir(exist_ok=True)
HOURLY_PARQUET = CACHE_DIR / 'hourly_long.parquet'

assert HOURLY_DIR.exists(), f'Dataset not found at {HOURLY_DIR.resolve()}'

def parse_sef_file(path: Path) -> pd.DataFrame:
    \"\"\"Parse one FAIRUrbTemp SEF TSV into long format (city, station_id, datetime, ta).

    station_id is the filename stem before '_AirTemp_' (handles 'D2195' and
    'AWEL_Blumenfeld'), Unicode-normalized to NFC so umlaut station names match
    the metadata TSVs. '-999' is a missing-value sentinel folded into NA.
    \"\"\"
    df = pd.read_csv(
        path, sep='\t', skiprows=12, na_values=['NA', '-999', '-999.0'],
        dtype={'Year':'Int64','Month':'Int64','Day':'Int64','Hour':'Int64','Minute':'Int64'},
        low_memory=False,
    )
    parts = df[['Year','Month','Day','Hour','Minute']].rename(columns=str.lower)
    return pd.DataFrame({
        'city': path.parent.name,
        'station_id': unicodedata.normalize('NFC', path.stem.split('_AirTemp_')[0]),
        'datetime': pd.to_datetime(parts),
        'ta': pd.to_numeric(df['Value'], errors='coerce'),
    })

if HOURLY_PARQUET.exists():
    print(f'Loading cached parquet {HOURLY_PARQUET} ...')
    hourly = pd.read_parquet(HOURLY_PARQUET)
else:
    print('Parsing SEF files (first run, ~1-2 min) ...')
    files = [f for c in sorted(HOURLY_DIR.iterdir()) if c.is_dir()
             for f in sorted(c.glob('*.tsv'))]
    hourly = pd.concat([parse_sef_file(f) for f in files], ignore_index=True)
    hourly['city'] = hourly['city'].astype('category')
    hourly['station_id'] = hourly['station_id'].astype('category')
    hourly.to_parquet(HOURLY_PARQUET, index=False)
    print(f'Saved {HOURLY_PARQUET} ({HOURLY_PARQUET.stat().st_size/1e6:.1f} MB)')

print(f'Loaded {len(hourly):,} rows, {hourly.station_id.nunique()} stations, '
      f'{hourly.city.nunique()} cities')
""")
```

Note: in the script the SEF docstring uses escaped `\"\"\"` and the tab is a literal `\t` inside the Python string — copy exactly as shown.

- [ ] **Step 2: Generate + execute** (standard commands)
Expected: output line `Loaded <N> rows, <M> stations, <K> cities` with no traceback.

- [ ] **Step 3: Verify station/city counts via a temporary assertion**

Run:
```bash
cd deliverable_3 && python -c "
import json; nb=json.load(open('d3.ipynb'))
out=''.join(o.get('text','') for c in nb['cells'] if c['cell_type']=='code' for o in c.get('outputs',[]) if o.get('output_type')=='stream')
print(out)
"
```
Expected: a "Loaded ... stations" line is present (full 375-after-filter count is asserted in Task 3 once filtering is applied).

- [ ] **Step 4: Integrity scan** — Expected: `errors=0 section-symbols=0`.

- [ ] **Step 5: Commit**
```bash
git add deliverable_3/build_d3.py deliverable_3/d3.ipynb
git commit -m "feat(d3): self-contained SEF loader + Section 2 data description"
```

---

## Task 3: Risk metric + both targets (Section 3.1), counts asserted

**Files:**
- Modify: `deliverable_3/build_d3.py` (append Section 3 intro + 3.1 markdown + risk/target code)

**Interfaces:**
- Consumes: `hourly`.
- Produces: DataFrame `risk` with columns `city, station_id, p95, is_hotspot_absolute, is_local_hotspot`.

- [ ] **Step 1: Append Section 3 intro, Section 3.1 markdown, and the risk-metric code cell**

```python
md(r"""
## 3. From raw records to a modeling table

We now turn millions of hourly readings into one row per station with a heat
measure and the features we will model. Every transformation here is
reproducible and explained, because the quality of this step decides whether
the later model means anything.
""")

md(r"""
### 3.1 Heat measure and the two definitions of "hotspot"

For each station we summarise its summers (June to August, 2019-2022) with the
**95th percentile of hourly air temperature** (`p95`). The 95th percentile
captures how hot a station gets during its hottest hours, while ignoring a
single freak reading that a plain maximum would over-react to. We keep only
stations with at least 50% of the possible summer hours, so a sparse station
cannot produce a noisy percentile.

We then label "hotspots" two different ways, and the contrast between them is
the analytical heart of this project:

- **Absolute hotspot** — the station's `p95` is in the top quartile across *all*
  stations in *all* cities. This is dominated by climate: a warm city's stations
  are hot in absolute terms.
- **Within-city (local) hotspot** — the station's `p95` is in the top quartile
  *inside its own city*. This removes the city's background climate and isolates
  **microclimate**: which spots run hot relative to their neighbours.

A planner needs both. The absolute view says which cities run hottest; the
within-city view says where the local hotspots sit inside a given city.
""")

code(r"""
STUDY_YEARS = [2019, 2020, 2021, 2022]
SUMMER_MONTHS = [6, 7, 8]
MAX_OBS = len(STUDY_YEARS) * 92 * 24    # theoretical max summer hours
MIN_AVAILABILITY = 0.50

m = (hourly['datetime'].dt.year.isin(STUDY_YEARS)
     & hourly['datetime'].dt.month.isin(SUMMER_MONTHS))
summer = hourly.loc[m].dropna(subset=['ta']).copy()

n_obs = summer.groupby('station_id', observed=True).size()
keep = n_obs[n_obs >= MIN_AVAILABILITY * MAX_OBS].index
summer = summer[summer['station_id'].isin(keep)]

risk = (summer.groupby(['city', 'station_id'], observed=True)['ta']
        .quantile(0.95).reset_index(name='p95'))

risk['is_hotspot_absolute'] = (risk['p95'] >= risk['p95'].quantile(0.75)).astype(int)
risk['is_local_hotspot'] = (
    risk.groupby('city', observed=True)['p95']
        .transform(lambda s: (s >= s.quantile(0.75)).astype(int))
)

print(f'Stations after availability filter: {len(risk)} across {risk.city.nunique()} cities')
print(f'Absolute hotspot rate: {risk.is_hotspot_absolute.mean():.0%}')
print(f'Local hotspot rate:    {risk.is_local_hotspot.mean():.0%}')
print('\nAbsolute hotspots by city (expect Zurich-dominated):')
print(risk[risk.is_hotspot_absolute == 1].city.value_counts().to_string())
print('\nLocal hotspots by city (expect spread across cities):')
print(risk[risk.is_local_hotspot == 1].city.value_counts().to_string())

assert 350 <= len(risk) <= 400, f'unexpected station count {len(risk)}'
for col in ['is_hotspot_absolute', 'is_local_hotspot']:
    assert 0.20 <= risk[col].mean() <= 0.30, f'{col} rate off'
""")
```

- [ ] **Step 2: Generate + execute** (standard commands). Expected: no traceback; both asserts pass.
- [ ] **Step 3: Confirm counts** — read stdout (as in Task 2 Step 3); expect "Stations after availability filter: 375 across 6 cities" (or within 350-400 if the archive differs).
- [ ] **Step 4: Integrity scan** — `errors=0 section-symbols=0`.
- [ ] **Step 5: Commit**
```bash
git add deliverable_3/build_d3.py deliverable_3/d3.ipynb
git commit -m "feat(d3): summer p95 risk metric + absolute & within-city targets"
```

---

## Task 4: Base features + NEW spatial features (Section 3.2), columns verified

**Files:**
- Modify: `deliverable_3/build_d3.py` (append Section 3.2 markdown + metadata-merge code + new-feature code)

**Interfaces:**
- Consumes: `risk`.
- Produces: DataFrame `df` with all model columns. Defines module-level lists `NUM_FEATURES_BASE`, `NUM_FEATURES_SPATIAL`, `NUM_FEATURES` (= base + spatial), `CAT_FEATURES = ['city']`, `FEATURES = NUM_FEATURES + CAT_FEATURES`. Defines `haversine_km(lat1,lon1,lat2,lon2)`. New columns: `distance_to_centroid`, `sensor_height_parsed`, `station_density_1km`, `knn_p95_k5`, `bearing_sin`, `bearing_cos`, `altitude_m`.

- [ ] **Step 1: Append Section 3.2 markdown and the metadata + base-feature code (ported from d2.ipynb cell 10)**

```python
md(r"""
### 3.2 Station context features

A station's heat is driven by where it sits. We attach each station's
coordinates and sensor height from the metadata tables (normalising ids to NFC
again so umlaut stations match), then engineer features that describe a
station's position. All of these come only from geography and metadata, never
from the temperature record, so they cannot leak the answer into the model.
""")

code(r"""
station_meta = pd.concat(
    [pd.read_csv(p, sep='\t') for p in sorted(META_DIR.glob('*/station_metadata_*.tsv'))],
    ignore_index=True,
)
station_meta['station_id'] = station_meta['station_id'].map(
    lambda s: unicodedata.normalize('NFC', str(s))
)

def parse_height(v):
    \"\"\"'4 m' -> 4.0 ; '2-3 m' -> 2.5 ; 2.0 -> 2.0 ; junk -> NaN\"\"\"
    if pd.isna(v): return np.nan
    s = str(v).lower().replace('m', '').strip()
    if '-' in s:
        a, b = s.split('-', 1)
        try: return (float(a) + float(b)) / 2
        except ValueError: return np.nan
    try: return float(s)
    except ValueError: return np.nan

station_meta['sensor_height_parsed'] = station_meta['sensor_height_m'].apply(parse_height)

meta_cols = ['station_id', 'latitude', 'longitude', 'sensor_height_parsed']
if 'altitude_m' in station_meta.columns:
    meta_cols.append('altitude_m')
df = risk.merge(station_meta[meta_cols], on='station_id', how='left')
if 'altitude_m' not in df.columns:
    df['altitude_m'] = np.nan

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1); dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat/2)**2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
    return 2 * R * np.arcsin(np.sqrt(a))

cent = (df.groupby('city', observed=True)[['latitude', 'longitude']]
          .mean().rename(columns={'latitude': 'clat', 'longitude': 'clon'}))
df = df.merge(cent, on='city')
df['distance_to_centroid'] = haversine_km(df.latitude, df.longitude, df.clat, df.clon)
df['city'] = df['city'].astype(str)

assert df[['latitude', 'longitude']].notna().all().all(), \
    'some stations have no coordinates -> station_id mismatch with metadata'
print(f'All {len(df)} stations matched metadata coordinates.')
""")
```

- [ ] **Step 2: Append Section 3.2 (new features) markdown + the spatial-feature code**

The new spatial features are computed here as plain columns for EDA and for the
final-fit feature table. The kNN-neighbour p95 is a leakage risk, so it is
recomputed **inside CV** in Task 7 via a custom estimator; the column built here
is for EDA and for the full-data fit only, and this is stated in the markdown.

```python
md(r"""
**New spatial features (Deliverable 3).** Distance to the city centre is a coarse
position signal. We add three richer ones, all derived from the coordinates we
already hold:

- **Local station density** counts how many other stations lie within 1 km. A
  dense cluster of sensors usually marks a built-up core, a proxy for the kind of
  place that traps heat.
- **Neighbourhood heat (kNN p95)** is the average `p95` of the five nearest
  *other* stations. It asks whether a station sits in a hot pocket. Because this
  feature is built from temperatures, using it carelessly would leak the answer,
  so when we validate we recompute it inside each training fold (Section 6). The
  column built here is for exploration and the full-data view only.
- **Direction from the centre** (the compass bearing, split into its sine and
  cosine) lets the model learn that, say, the south side of a city runs hotter,
  which plain distance cannot express.

We also keep **elevation** where the metadata provides it; it is populated for
only a few cities, so it acts as a city-specific signal rather than a universal
one.
""")

code(r"""
coords = df[['latitude', 'longitude']].to_numpy()
n = len(df)

# pairwise great-circle distances (n is small, ~375, so a dense matrix is fine)
dist_km = np.zeros((n, n))
for i in range(n):
    dist_km[i] = haversine_km(coords[i, 0], coords[i, 1], coords[:, 0], coords[:, 1])

# Local station density: other stations within 1 km (exclude self on the diagonal)
within_1km = (dist_km <= 1.0)
np.fill_diagonal(within_1km, False)
df['station_density_1km'] = within_1km.sum(axis=1)

# kNN-neighbour p95 (k=5 nearest OTHER stations), full-data version for EDA only
K = 5
p95_vals = df['p95'].to_numpy()
knn_p95 = np.empty(n)
for i in range(n):
    order = np.argsort(dist_km[i])
    order = order[order != i][:K]
    knn_p95[i] = p95_vals[order].mean()
df['knn_p95_k5'] = knn_p95

# Direction from the city centroid, as sine/cosine of the bearing
dlon = np.radians(df['longitude'] - df['clon'])
lat1 = np.radians(df['clat']); lat2 = np.radians(df['latitude'])
x = np.sin(dlon) * np.cos(lat2)
y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
bearing = np.arctan2(x, y)
df['bearing_sin'] = np.sin(bearing)
df['bearing_cos'] = np.cos(bearing)

NUM_FEATURES_BASE = ['latitude', 'longitude', 'distance_to_centroid', 'sensor_height_parsed']
NUM_FEATURES_SPATIAL = ['station_density_1km', 'bearing_sin', 'bearing_cos', 'altitude_m']
NUM_FEATURES = NUM_FEATURES_BASE + NUM_FEATURES_SPATIAL
CAT_FEATURES = ['city']
FEATURES = NUM_FEATURES + CAT_FEATURES

print('New spatial features built. Summary:')
print(df[['station_density_1km', 'knn_p95_k5', 'bearing_sin', 'bearing_cos']].describe().round(2))
print('\naltitude_m non-missing by city:')
print(df.groupby('city')['altitude_m'].apply(lambda s: s.notna().sum()).to_string())
assert df['station_density_1km'].min() >= 0
assert df[['bearing_sin', 'bearing_cos']].abs().max().max() <= 1.0 + 1e-9
""")
```

Note on `knn_p95_k5`: it is intentionally **excluded** from `NUM_FEATURES` (the model feature list) because its leak-free form is injected inside CV in Task 7. It stays in `df` for EDA only.

- [ ] **Step 3: Generate + execute** (standard commands). Expected: no traceback; the two asserts pass; density/bearing summary prints.
- [ ] **Step 4: Integrity scan** — `errors=0 section-symbols=0`.
- [ ] **Step 5: Commit**
```bash
git add deliverable_3/build_d3.py deliverable_3/d3.ipynb
git commit -m "feat(d3): base + new spatial features (density, kNN p95, bearing, elevation)"
```

---

## Task 5: Preprocessing pipeline + leak-free kNN transformer (Section 3.3)

**Files:**
- Modify: `deliverable_3/build_d3.py` (append Section 3.3 markdown + pipeline code + kNN transformer)

**Interfaces:**
- Consumes: `df`, `NUM_FEATURES`, `CAT_FEATURES`, `FEATURES`, `haversine_km`.
- Produces: `make_pipeline(classifier, add_knn=False)` returning a leak-free sklearn `Pipeline`; a `KNNNeighbourP95` transformer class; model matrices `X = df[FEATURES + ['latitude','longitude','p95']]`-style is NOT used — instead `X = df[FEATURES].copy()`, `y_abs`, `y_loc`. Defines `X_geo` (base features only) and `X_full` (base+spatial) for the ablation in Task 7.

- [ ] **Step 1: Append Section 3.3 markdown + pipeline code**

The leak-free kNN feature is implemented as a transformer that, given the train
rows' coordinates and p95, computes each row's neighbour-mean p95 using only the
training set's neighbours. To keep the data flowing through the `ColumnTransformer`
simple, the transformer reads coordinates and p95 from columns carried alongside
the features and appends one new column.

```python
md(r"""
### 3.3 A leak-free preprocessing pipeline

Everything the model learns from the data must be learned on the training rows
only, never the rows we test on. We wrap preprocessing in a scikit-learn
`Pipeline` so that during cross-validation each step is fit on the training fold
and merely applied to the held-out fold. The numeric features are median-imputed
(the parsed sensor height and elevation have gaps) and standardised; the city is
left out of the model as a predictor because the within-city target already has
the city removed, and feeding the city back in would be circular. We keep `city`
only as a grouping key for the cross-city test in Section 6.

The neighbourhood-heat feature needs special care: its value for a station
depends on other stations' temperatures, so if a test station's neighbours
include training stations we must compute it using training neighbours only. We
implement it as a transformer that stores the training coordinates and p95 when
it is fit, and computes neighbour averages for any rows it later transforms.
""")

code(r"""
from sklearn.base import BaseEstimator, TransformerMixin

class KNNNeighbourP95(BaseEstimator, TransformerMixin):
    \"\"\"Append mean p95 of the k nearest TRAINING stations (leak-free).

    Expects input columns ['latitude','longitude','p95']; outputs one column.
    On fit it memorises training coords+p95; on transform it looks up neighbours
    among the training stations only.\"\"\"
    def __init__(self, k=5):
        self.k = k
    def fit(self, X, y=None):
        A = np.asarray(X, dtype=float)
        self.train_lat_, self.train_lon_, self.train_p95_ = A[:, 0], A[:, 1], A[:, 2]
        return self
    def transform(self, X):
        A = np.asarray(X, dtype=float)
        out = np.empty(len(A))
        for i in range(len(A)):
            d = haversine_km(A[i, 0], A[i, 1], self.train_lat_, self.train_lon_)
            order = np.argsort(d)
            # drop exact-self matches (distance 0) so a train row uses its neighbours
            order = order[d[order] > 1e-9][:self.k]
            out[i] = self.train_p95_[order].mean()
        return out.reshape(-1, 1)

def make_pipeline(classifier, add_knn=False):
    \"\"\"Leak-free ColumnTransformer (+ optional kNN p95) + classifier.\"\"\"
    transformers = [
        ('num', Pipeline([('impute', SimpleImputer(strategy='median')),
                          ('scale', StandardScaler())]), NUM_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CAT_FEATURES),
    ]
    if add_knn:
        transformers.append(('knn', Pipeline([
            ('nbr', KNNNeighbourP95(k=5)),
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
        ]), ['latitude', 'longitude', 'p95']))
    pre = ColumnTransformer(transformers)
    return Pipeline([('pre', pre), ('clf', classifier)])

# Feature matrices. X carries p95 ONLY so the kNN transformer can read it; the
# raw p95 column is never exposed to the numeric block (NUM_FEATURES excludes it).
X = df[FEATURES + ['p95']].copy()
y_abs = df['is_hotspot_absolute'].copy()
y_loc = df['is_local_hotspot'].copy()
groups = df['city'].to_numpy()

# smoke test (not evaluation)
_smoke = make_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), add_knn=True)
_smoke.fit(X, y_abs)
assert _smoke.predict_proba(X).shape == (len(X), 2)
print('Pipeline smoke test OK (with leak-free kNN p95 feature).')
""")
```

Note: `NUM_FEATURES` excludes `p95` and `knn_p95_k5`, so carrying `p95` in `X` does not leak — only the `KNNNeighbourP95` transformer (fit on train only) consumes it.

- [ ] **Step 2: Generate + execute**. Expected: `Pipeline smoke test OK ...`, no traceback.
- [ ] **Step 3: Integrity scan** — `errors=0 section-symbols=0`.
- [ ] **Step 4: Commit**
```bash
git add deliverable_3/build_d3.py deliverable_3/d3.ipynb
git commit -m "feat(d3): leak-free preprocessing pipeline + kNN-neighbour-p95 transformer"
```

---

## Task 6: EDA (Section 4) — two motivating findings + feature views

**Files:**
- Modify: `deliverable_3/build_d3.py` (append Section 4 markdown + 2-3 plotting code cells)

**Interfaces:**
- Consumes: `df`, `risk`.
- Produces: figures only (no new variables required downstream).

- [ ] **Step 1: Append Section 4 intro + EDA plot cells**

```python
md(r"""
## 4. Exploratory data analysis

Before modeling we look at the data to understand the patterns the model will
have to capture, and to motivate the two-target design. Two findings stand out
and shape the rest of the notebook.
""")

md(r"""
### 4.1 Climate dominates the absolute ranking

If we rank stations by raw `p95`, the hottest are concentrated in the warmest
cities. The boxplot below shows each city's spread of station `p95` values. The
cities sit at clearly different levels, so an absolute "top quartile" label is
mostly a statement about which city a station is in.
""")

code(r"""
order = df.groupby('city')['p95'].median().sort_values(ascending=False).index
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df, x='city', y='p95', order=order, ax=ax)
ax.set_xlabel('City'); ax.set_ylabel('Summer p95 air temperature (degC)')
ax.set_title('Station heat (summer p95) by city — cities sit at different climate levels')
plt.xticks(rotation=20); plt.tight_layout(); plt.show()
""")

md(r"""
### 4.2 Microclimate exists within every city

Within a single city, stations still differ by several degrees. That within-city
spread is the microclimate signal our local-hotspot target tries to predict. The
plot below shows, for each city, how far its hottest stations sit above its
coolest, confirming there is a real within-city range to explain.
""")

code(r"""
spread = (df.groupby('city')['p95'].agg(['min', 'max'])
            .assign(range=lambda d: d['max'] - d['min'])
            .sort_values('range', ascending=False))
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(x=spread.index, y=spread['range'], ax=ax)
ax.set_xlabel('City'); ax.set_ylabel('Within-city p95 range (degC)')
ax.set_title('Hottest minus coolest station within each city — the microclimate range')
plt.xticks(rotation=20); plt.tight_layout(); plt.show()
print(spread.round(2).to_string())
""")

md(r"""
### 4.3 Do the new spatial features track heat?

A quick check that the engineered features carry signal worth modeling. We plot
neighbourhood heat (mean p95 of the five nearest stations) against a station's
own p95; a positive trend means heat is spatially clustered, which is what the
feature is meant to capture.
""")

code(r"""
fig, ax = plt.subplots(figsize=(7, 6))
sns.scatterplot(data=df, x='knn_p95_k5', y='p95', hue='city', ax=ax, alpha=0.7)
ax.set_xlabel('Neighbourhood heat: mean p95 of 5 nearest stations (degC)')
ax.set_ylabel('Station summer p95 (degC)')
ax.set_title('Heat is spatially clustered: a station resembles its neighbours')
ax.legend(title='City', fontsize=8, loc='best')
plt.tight_layout(); plt.show()
corr = df['knn_p95_k5'].corr(df['p95'])
print(f'Correlation(neighbour p95, own p95) = {corr:.2f}')
""")
```

- [ ] **Step 2: Generate + execute**. Expected: three figures produced; correlation prints; no traceback.
- [ ] **Step 3: Verify figures exist**
```bash
cd deliverable_3 && python -c "
import json; nb=json.load(open('d3.ipynb'))
imgs=sum(1 for c in nb['cells'] if c['cell_type']=='code' for o in c.get('outputs',[]) if o.get('output_type')=='display_data' and 'image/png' in o.get('data',{}))
print('figures:', imgs)
"
```
Expected: `figures: 3` (or more).
- [ ] **Step 4: Integrity scan** — `errors=0 section-symbols=0`.
- [ ] **Step 5: Commit**
```bash
git add deliverable_3/build_d3.py deliverable_3/d3.ipynb
git commit -m "feat(d3): EDA — climate dominance, within-city spread, spatial clustering"
```

---

## Task 7: Models + StratifiedKFold contrast WITH feature ablation (Sections 5 + 6.1)

**Files:**
- Modify: `deliverable_3/build_d3.py` (append Section 5 markdown + model-fit code; Section 6.1 markdown + CV contrast code)

**Interfaces:**
- Consumes: `make_pipeline`, `X`, `y_abs`, `y_loc`, `groups`, `NUM_FEATURES_BASE`.
- Produces: fitted `logit`, `rf` (GridSearchCV on absolute target); dict `proba[(target, model, featureset)]`; DataFrame `contrast` (MultiIndex target/model/featureset with ROC-AUC, PR-AUC). `best_name`, `p_best` (the working absolute RF out-of-fold probabilities, full feature set) for Section 6.2-6.4.

- [ ] **Step 1: Append Section 5 markdown + model definitions (ported/adapted from d2 cells 13-15)**

```python
md(r"""
## 5. Modeling approach

The screening question is naturally a **classification** problem: given a
station's location and context, is it a hotspot? Classification also gives us
the evaluation tools a screening decision needs, above all the ability to tune
how aggressively we flag stations. We build a ladder of models, from a trivial
baseline to an interpretable linear model to a flexible non-linear one, and we
run the whole ladder on **both** targets so the absolute-versus-within-city
contrast is built into the design rather than added as an afterthought.

We expect only modest skill from geography alone, and that expectation is itself
a finding: the true drivers of microclimate (land cover, building density,
surface sealing) are not in this dataset. A simple model that we can explain is
worth more here than a complex one we cannot.
""")

code(r"""
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
dummy = make_pipeline(DummyClassifier(strategy='stratified', random_state=RANDOM_STATE), add_knn=True)
logit = GridSearchCV(
    make_pipeline(LogisticRegression(class_weight='balanced', max_iter=1000,
                                     random_state=RANDOM_STATE), add_knn=True),
    param_grid={'clf__C': [0.1, 1.0, 10.0]},
    scoring='average_precision', cv=cv, n_jobs=-1,
)
rf = GridSearchCV(
    make_pipeline(RandomForestClassifier(class_weight='balanced',
                                         random_state=RANDOM_STATE, n_jobs=-1), add_knn=True),
    param_grid={'clf__n_estimators': [200, 400], 'clf__max_depth': [3, 5, None]},
    scoring='average_precision', cv=cv, n_jobs=-1,
)
logit.fit(X, y_abs); rf.fit(X, y_abs)
print('LogReg best params:', logit.best_params_)
print('RF best params:    ', rf.best_params_)
""")
```

- [ ] **Step 2: Append Section 6 intro + 6.1 markdown + the contrast/ablation CV code**

This is the core result. It runs both targets, both the geography-only feature
set and the enriched set, so we can see whether the new spatial features rescue
the within-city model.

```python
md(r"""
## 6. Validation and results

We judge every model by cross-validation: the data are split into five folds,
each fold is predicted by a model trained on the other four, and we score the
out-of-fold predictions. This simulates predicting stations the model has not
seen. We report ROC-AUC (ranking quality across all thresholds) and PR-AUC
(precision against the minority hotspot class, the metric that matters when
positives are rare). A no-skill model scores about 0.5 on ROC-AUC.
""")

md(r"""
### 6.1 The contrast, and whether new features change it

The table below crosses the two targets with the two feature sets. The D2
finding was that the **absolute** target is predictable from geography while the
**within-city** target is not. Here we ask whether the new spatial features
(neighbourhood heat, density, direction, elevation) move the within-city number
off chance. Whatever the answer, it is informative: a lift means microclimate is
partly recoverable from spatial structure; no lift confirms that the missing
ingredient is data we do not have.
""")

code(r"""
# geography-only uses the base numeric features; enriched uses all + leak-free kNN
def oof(estimator, yt):
    cv_t = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(estimator, X, yt, cv=cv_t, method='predict_proba')[:, 1]

def pipe_geo(clf):
    pre = ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median')),
                          ('scale', StandardScaler())]), NUM_FEATURES_BASE),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CAT_FEATURES),
    ])
    return Pipeline([('pre', pre), ('clf', clf)])

def fresh_models(add_knn):
    mk = (lambda clf: make_pipeline(clf, add_knn=True)) if add_knn else pipe_geo
    return {
        'Dummy (stratified)': mk(DummyClassifier(strategy='stratified', random_state=RANDOM_STATE)),
        'LogReg (balanced)':  mk(LogisticRegression(class_weight='balanced', max_iter=1000,
                                                    C=logit.best_params_['clf__C'], random_state=RANDOM_STATE)),
        'RandomForest':       mk(RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE,
                                                        n_jobs=-1, **{k.replace('clf__',''):v for k,v in rf.best_params_.items()})),
    }

targets = {'absolute': y_abs, 'within-city': y_loc}
proba, rows = {}, []
for fs_name, add_knn in [('geography-only', False), ('enriched', True)]:
    for tname, yt in targets.items():
        for mname, est in fresh_models(add_knn).items():
            p = oof(est, yt)
            proba[(tname, mname, fs_name)] = p
            rows.append({'target': tname, 'features': fs_name, 'model': mname,
                         'ROC-AUC': roc_auc_score(yt, p),
                         'PR-AUC': average_precision_score(yt, p)})
contrast = pd.DataFrame(rows).set_index(['target', 'features', 'model']).round(3)
print(contrast)

p_best = proba[('absolute', 'RandomForest', 'enriched')]
best_name = 'RandomForest (absolute, enriched)'
abs_rf = contrast.loc[('absolute', 'enriched', 'RandomForest'), 'ROC-AUC']
loc_rf = contrast.loc[('within-city', 'enriched', 'RandomForest'), 'ROC-AUC']
print(f'\nAbsolute RF ROC-AUC (enriched): {abs_rf:.3f}')
print(f'Within-city RF ROC-AUC (enriched): {loc_rf:.3f}')
assert abs_rf > 0.60, 'absolute model lost its skill — investigate'
""")
```

- [ ] **Step 2 (run): Generate + execute**. Expected: the `contrast` table prints with 12 rows; assert passes. Record `abs_rf` and `loc_rf` for Task 12.
- [ ] **Step 3: Integrity scan** — `errors=0 section-symbols=0`.
- [ ] **Step 4: Commit**
```bash
git add deliverable_3/build_d3.py deliverable_3/d3.ipynb
git commit -m "feat(d3): baseline->RF ladder, two targets x two feature sets (the contrast + ablation)"
```

---

## Task 8: ROC/PR + threshold tuning + error analysis (Sections 6.2-6.3)

**Files:**
- Modify: `deliverable_3/build_d3.py` (append 6.2 ROC/PR markdown+code; 6.3 threshold markdown+code; error-analysis code)

**Interfaces:**
- Consumes: `p_best`, `y_abs`, `df`.
- Produces: `t_star` (chosen threshold), `yhat_star`; a misclassified-stations table.

- [ ] **Step 1: Append 6.2 (ROC/PR curves) markdown + code**

```python
md(r"""
### 6.2 ROC and precision-recall curves (working model)

We focus the operating-point analysis on the model that actually has skill: the
absolute-target RandomForest with the enriched features. The ROC curve shows the
trade-off between catching true hotspots and raising false alarms across all
thresholds; the precision-recall curve is the more honest view here because
hotspots are the minority class.
""")

code(r"""
fpr, tpr, _ = roc_curve(y_abs, p_best)
prec, rec, _ = precision_recall_curve(y_abs, p_best)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_abs, p_best):.3f}')
axes[0].plot([0, 1], [0, 1], '--', color='grey')
axes[0].set_xlabel('False positive rate'); axes[0].set_ylabel('True positive rate')
axes[0].set_title('ROC — working model (absolute, enriched)'); axes[0].legend()
axes[1].plot(rec, prec, label=f'PR-AUC = {average_precision_score(y_abs, p_best):.3f}')
axes[1].axhline(y_abs.mean(), ls='--', color='grey', label=f'baseline = {y_abs.mean():.2f}')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall — working model'); axes[1].legend()
plt.tight_layout(); plt.show()
""")
```

- [ ] **Step 2: Append 6.3 (threshold tuning) markdown + code (ported/adapted from d2 cell 22)**

```python
md(r"""
### 6.3 Choosing the screening threshold

The default 0.5 cut-off is rarely the right screening decision. Missing a real
hotspot (a false negative) means a genuinely hot area goes unprotected, which we
treat as more costly than a false alarm (wasting a site visit). We weight a miss
three times a false alarm and pick the threshold that minimises the total cost.
This makes the screening posture explicit rather than an accident of the default.
""")

code(r"""
COST_FN, COST_FP = 3, 1
def metrics_at(yt, p, thr):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yh).ravel()
    return {'thr': thr, 'precision': precision_score(yt, yh, zero_division=0),
            'recall': recall_score(yt, yh, zero_division=0),
            'f1': f1_score(yt, yh, zero_division=0),
            'cost': COST_FN * fn + COST_FP * fp}
grid = np.linspace(0.05, 0.95, 91)
sweep = pd.DataFrame([metrics_at(y_abs, p_best, t) for t in grid])
t_star = float(sweep.loc[sweep['cost'].idxmin(), 'thr'])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(sweep['thr'], sweep['precision'], label='Precision')
ax.plot(sweep['thr'], sweep['recall'], label='Recall')
ax.plot(sweep['thr'], sweep['f1'], label='F1')
ax.axvline(t_star, ls='--', color='red', label=f'chosen threshold = {t_star:.2f}')
ax.set_xlabel('Decision threshold (predicted probability)')
ax.set_ylabel('Score'); ax.set_title('Threshold sweep — chosen at minimum expected cost (FN:FP = 3:1)')
ax.legend(); plt.tight_layout(); plt.show()
print(f'Chosen threshold: {t_star:.2f}')
print(pd.DataFrame([metrics_at(y_abs, p_best, t_star)]).round(3).to_string(index=False))
""")
```

- [ ] **Step 3: Append 6.3 error-analysis markdown + code (ported/adapted from d2 cell 24)**

```python
md(r"""
At the chosen threshold we inspect the stations the model gets wrong. If the
false negatives and false positives share traits — sitting at the city edge, an
unusual sensor height, one particular city — that tells us where the screen is
weakest and what a human reviewer should double-check.
""")

code(r"""
yhat_star = (p_best >= t_star).astype(int)
ConfusionMatrixDisplay(confusion_matrix(y_abs, yhat_star),
                       display_labels=['not hotspot', 'hotspot']).plot(cmap='Blues')
plt.title(f'Confusion matrix — working model at threshold {t_star:.2f}')
plt.tight_layout(); plt.show()

err = df.copy()
err['y_true'] = y_abs.to_numpy(); err['y_pred'] = yhat_star; err['p_hotspot'] = p_best
wrong = err[err['y_true'] != err['y_pred']]
show = (wrong[['city', 'p95', 'distance_to_centroid', 'sensor_height_parsed', 'p_hotspot', 'y_true']]
        .rename(columns={'p95': 'p95 (degC)', 'distance_to_centroid': 'distance to centroid (km)',
                         'sensor_height_parsed': 'sensor height (m)', 'p_hotspot': 'P(hotspot)'}))
print(f'{len(wrong)} misclassified stations at threshold {t_star:.2f}:')
print(show.round(2).to_string(index=False))
print('\nMisclassifications by city:'); print(wrong['city'].value_counts().to_string())
""")
```

- [ ] **Step 4: Generate + execute**. Expected: ROC/PR figure, threshold figure, confusion matrix; `t_star` prints; no traceback.
- [ ] **Step 5: Integrity scan** — `errors=0 section-symbols=0`.
- [ ] **Step 6: Commit**
```bash
git add deliverable_3/build_d3.py deliverable_3/d3.ipynb
git commit -m "feat(d3): ROC/PR curves, cost-based threshold, error analysis"
```

---

## Task 9: Cross-city transfer (GroupKFold) — Section 6.4

**Files:**
- Modify: `deliverable_3/build_d3.py` (append 6.4 markdown + GroupKFold code)

**Interfaces:**
- Consumes: `make_pipeline`, `X`, `y_abs`, `groups`, `df`.
- Produces: DataFrame `transfer` (per-held-out-city ROC-AUC); `transfer_mean`.

- [ ] **Step 1: Append 6.4 markdown + code**

```python
md(r"""
### 6.4 Can the model transfer to a city it has never seen?

Cross-validation above mixes all six cities into every fold, so the model always
trains on some stations from each city. A screening tool is more useful if it
works in a city with no training data at all. We test this with leave-one-city-out
validation: train on five cities, predict the sixth, and repeat. We score the
absolute target, the one with real skill. A large drop from the mixed-fold score
would mean the model leans on city-specific quirks rather than a transferable
pattern. Two cities (Amsterdam, Berlin) have only ten stations each, so their
held-out scores are noisy; we report every city and read the pattern, not any
single number.
""")

code(r"""
gkf = GroupKFold(n_splits=df['city'].nunique())
rows = []
for tr, te in gkf.split(X, y_abs, groups):
    held = df.iloc[te]['city'].iloc[0]
    est = make_pipeline(RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE,
                                               n_jobs=-1,
                                               **{k.replace('clf__',''):v for k,v in rf.best_params_.items()}),
                        add_knn=True)
    est.fit(X.iloc[tr], y_abs.iloc[tr])
    p = est.predict_proba(X.iloc[te])[:, 1]
    yt = y_abs.iloc[te]
    auc = roc_auc_score(yt, p) if yt.nunique() > 1 else np.nan
    rows.append({'held_out_city': held, 'n_stations': len(te),
                 'n_positive': int(yt.sum()), 'ROC-AUC': auc})
transfer = pd.DataFrame(rows).sort_values('n_stations', ascending=False)
transfer_mean = transfer['ROC-AUC'].mean()
print(transfer.round(3).to_string(index=False))
print(f'\nMean leave-one-city-out ROC-AUC: {transfer_mean:.3f}')
print(f'(Compare to mixed-fold absolute RF ROC-AUC computed in Section 6.1.)')
""")
```

- [ ] **Step 2: Generate + execute**. Expected: per-city table prints with 6 rows; mean prints; no traceback.
- [ ] **Step 3: Integrity scan** — `errors=0 section-symbols=0`.
- [ ] **Step 4: Commit**
```bash
git add deliverable_3/build_d3.py deliverable_3/d3.ipynb
git commit -m "feat(d3): leave-one-city-out cross-city transfer test"
```

---

## Task 10: Learning curves — Section 6.5

**Files:**
- Modify: `deliverable_3/build_d3.py` (append 6.5 markdown + learning_curve code)

**Interfaces:**
- Consumes: `make_pipeline`, `X`, `y_abs`, `rf.best_params_`.
- Produces: figure; arrays `train_scores`, `val_scores` (for prose, not downstream).

- [ ] **Step 1: Append 6.5 markdown + code**

```python
md(r"""
### 6.5 Would more data or better features help more?

A learning curve trains the model on increasing fractions of the data and plots
the cross-validated score. If the validation score is still climbing at the full
sample, more stations would help. If the two curves have met and flattened at a
modest level, the model has learned what it can from these features and the limit
is the features themselves — more rows would not help, but better inputs (land
cover, density) would. This diagnosis is the evidence behind our recommendation
in Section 8.
""")

code(r"""
lc_est = make_pipeline(RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE,
                                              n_jobs=-1,
                                              **{k.replace('clf__',''):v for k,v in rf.best_params_.items()}),
                       add_knn=True)
sizes, train_scores, val_scores = learning_curve(
    lc_est, X, y_abs, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
    scoring='roc_auc', train_sizes=np.linspace(0.2, 1.0, 6), n_jobs=-1,
)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(sizes, train_scores.mean(1), 'o-', label='Training ROC-AUC')
ax.plot(sizes, val_scores.mean(1), 'o-', label='Cross-validated ROC-AUC')
ax.fill_between(sizes, val_scores.mean(1)-val_scores.std(1), val_scores.mean(1)+val_scores.std(1), alpha=0.15)
ax.set_xlabel('Number of training stations'); ax.set_ylabel('ROC-AUC')
ax.set_title('Learning curve — working model (absolute target)')
ax.legend(); plt.tight_layout(); plt.show()
print(f'Train vs CV ROC-AUC at full size: {train_scores.mean(1)[-1]:.3f} vs {val_scores.mean(1)[-1]:.3f}')
""")
```

- [ ] **Step 2: Generate + execute**. Expected: learning-curve figure; final train/CV numbers print; no traceback.
- [ ] **Step 3: Integrity scan** — `errors=0 section-symbols=0`.
- [ ] **Step 4: Commit**
```bash
git add deliverable_3/build_d3.py deliverable_3/d3.ipynb
git commit -m "feat(d3): learning curve (bias vs variance diagnosis)"
```

---

## Task 11: Bootstrapped uncertainty + limitations — Section 7

**Files:**
- Modify: `deliverable_3/build_d3.py` (append Section 7 markdown + bootstrap code + limitations markdown)

**Interfaces:**
- Consumes: `proba`, `y_abs`, `y_loc`, `rng`.
- Produces: DataFrame `ci` (ROC-AUC point estimate + 95% CI for absolute and within-city working models); figure.

- [ ] **Step 1: Append Section 7 intro + bootstrap code**

```python
md(r"""
## 7. Uncertainty and limitations

A single ROC-AUC number hides how much it could shift if we had sampled
different stations. With only a few hundred stations this matters. We quantify it
by bootstrapping: we resample the stations with replacement many times, recompute
the score each time, and report the 2.5th-97.5th percentile range as a 95%
confidence interval. This lets us say whether the gap between the absolute model
and the within-city model is real or could be noise.
""")

code(r"""
def bootstrap_auc(yt, p, n_boot=2000):
    yt = np.asarray(yt); p = np.asarray(p); n = len(yt)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.unique(yt[idx]).size < 2:
            continue
        stats.append(roc_auc_score(yt[idx], p[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return np.mean(stats), lo, hi

rows = []
for tname, yt in [('absolute', y_abs), ('within-city', y_loc)]:
    p = proba[(tname, 'RandomForest', 'enriched')]
    mean, lo, hi = bootstrap_auc(yt, p)
    rows.append({'target': tname, 'ROC-AUC': roc_auc_score(yt, p),
                 'boot_mean': mean, 'CI_low': lo, 'CI_high': hi})
ci = pd.DataFrame(rows).round(3)
print(ci.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4))
ax.errorbar(ci['ROC-AUC'], ci['target'],
            xerr=[ci['ROC-AUC']-ci['CI_low'], ci['CI_high']-ci['ROC-AUC']],
            fmt='o', capsize=6)
ax.axvline(0.5, ls='--', color='grey', label='no-skill (0.5)')
ax.set_xlabel('ROC-AUC (95% bootstrap CI)'); ax.set_ylabel('')
ax.set_title('Absolute vs within-city: point estimate with uncertainty')
ax.legend(); plt.tight_layout(); plt.show()
""")
```

- [ ] **Step 2: Append Section 7 limitations markdown (prose, no numbers needed)**

```python
md(r"""
### 7.1 What this analysis can and cannot say

The confidence intervals frame two kinds of uncertainty. **Epistemic** uncertainty
is the part we could shrink with more or better data: the wide intervals come
partly from a small sample, and the modest scores come from features that only
proxy the real drivers. **Aleatoric** uncertainty is irreducible: a station's
heat depends on fine-scale surroundings that no amount of coordinates will fully
capture.

Honest limitations of the study:

- The features are thin. Latitude, longitude, distance, direction, density and
  sensor height are weak stand-ins for what actually drives microclimate, namely
  land cover, surface sealing and building density. We name this as the main
  reason the within-city signal is hard to predict.
- The within-city label is **relative**. A within-city hotspot in a cool city is
  not hot in absolute terms, so the two views must be read together.
- The sample is small (a few hundred stations, two cities with only ten), which
  widens every interval and makes the cross-city scores for small cities noisy.
- Climate is excluded from the within-city target by design, to isolate
  microclimate; that is a deliberate scope choice, not an oversight.
""")
```

- [ ] **Step 3: Generate + execute**. Expected: `ci` table prints with 2 rows + intervals; error-bar figure; no traceback.
- [ ] **Step 4: Integrity scan** — `errors=0 section-symbols=0`.
- [ ] **Step 5: Commit**
```bash
git add deliverable_3/build_d3.py deliverable_3/d3.ipynb
git commit -m "feat(d3): bootstrapped ROC-AUC CIs + limitations (aleatoric vs epistemic)"
```

---

## Task 12: Fill all interpretation prose with real numbers (Sections 1, 6.1, 8, 9)

**Files:**
- Modify: `deliverable_3/build_d3.py` (replace the Section 1 stub; add interpretation prose under 6.1/6.4/6.5; add Sections 8 and 9)

**Interfaces:**
- Consumes: executed-notebook stdout numbers from Tasks 3, 7, 8, 9, 10, 11.

- [ ] **Step 1: Read every computed number from the executed notebook**
```bash
cd deliverable_3 && python -c "
import json; nb=json.load(open('d3.ipynb'))
for c in nb['cells']:
    if c['cell_type']=='code':
        for o in c.get('outputs',[]):
            if o.get('output_type')=='stream': print(''.join(o.get('text','')), end='')
"
```
Record: station/city counts and hotspot rates (Task 3); the full `contrast` table incl. absolute vs within-city ROC-AUC for both feature sets (Task 7); `t_star` and its precision/recall/F1 (Task 8); cross-city `transfer` table + mean (Task 9); learning-curve final train/CV AUC (Task 10); bootstrap CIs (Task 11).

- [ ] **Step 2: Replace the Section 1 stub with the full motivation** (use real counts)

Replace the `## 1. Question and motivation` stub cell's body with elaborate prose covering: the public-health stakes of urban heat (heat stress and excess mortality among the elderly, children, and people with pre-existing conditions; street-level exposure varies block to block); the screening question in plain words; what "heat risk" means here (summer p95 of hourly `ta`); the two lenses (absolute vs within-city) and why a planner needs both; the dataset in one line (N stations across K cities, real counts from Task 3); and the success criterion (an honest, decision-useful screen, not a high accuracy number). No `:` / `-` connector overuse; short declarative sentences. No `§`.

- [ ] **Step 3: Add interpretation prose cells** after 6.1, 6.4, 6.5 stating the real numbers and what they mean (did the new features move the within-city score; how far the cross-city score dropped; what the learning curve says about data vs features).

- [ ] **Step 4: Add Section 8 (decision) and Section 9 (conclusion)**

```python
md(r"""
## 8. Decision-oriented interpretation and recommendation

`fill after running` — write a two-tier screening recommendation using the real
numbers: (1) use the absolute view to rank which cities run hottest and deserve
first attention; (2) inside a chosen city, use the working model at the threshold
t_star from Section 6.3 to shortlist candidate hotspots for a site visit, stating
the precision/recall trade-off in plain terms (at this threshold the screen
catches X% of true hotspots at Y% precision). State that the within-city model is
not yet reliable enough to deploy on its own, supported by the chance-level score
and its confidence interval. Close with the highest-value next step earned by the
learning curve: acquiring real land-cover / surface-sealing data, not collecting
more stations.
""")

md(r"""
## 9. Conclusion

`fill after running` — three to five sentences closing the arc: the question, the
two views of risk, the contrast (absolute is predictable, within-city is not from
geography alone), how far we trust it (cross-city transfer + confidence
intervals), and the single recommended action. No new claims beyond what the
notebook showed.
""")
```

Then immediately replace both `fill after running` blocks with the real-number prose (Sections 8 and 9 are prose-only, so they can be finalised in this same task).

- [ ] **Step 5: Generate + execute**. Expected: no traceback.
- [ ] **Step 6: Integrity scan** — Expected: `errors=0 section-symbols=0 fill-after-running=0`.
- [ ] **Step 7: Commit**
```bash
git add deliverable_3/build_d3.py deliverable_3/d3.ipynb
git commit -m "feat(d3): fill motivation, interpretation, decision, and conclusion prose"
```

---

## Task 13: Final clean-kernel run + full verification

**Files:** none (verification only).

- [ ] **Step 1: Restart-and-run-all from a clean kernel**
```bash
cd deliverable_3 && jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=cdap --ExecutePreprocessor.timeout=900 d3.ipynb
```
Expected: exit code 0, no traceback.

- [ ] **Step 2: Full integrity scan** (standard scan). Expected: `errors=0 section-symbols=0 fill-after-running=0`.

- [ ] **Step 3: Confirm self-contained + no forbidden content**
```bash
cd deliverable_3 && python -c "
import json; src=' '.join(''.join(c['source']) for c in json.load(open('d3.ipynb'))['cells'])
assert 'Sunny Tamrakar' not in src, 'author name must be Kriti Shrestha'
assert 'colab' not in src.lower() and 'kaggle' not in src.lower(), 'no Colab/Kaggle'
assert '../archive' in src, 'must read shared archive'
assert 'd1.ipynb' not in src and 'd2.ipynb' not in src, 'must be standalone'
print('standalone + naming + data-source checks passed')
"
```
Expected: `standalone + naming + data-source checks passed`.

- [ ] **Step 4: Reload reminder + final commit**

Remind the user to reload `d3.ipynb` in their IDE (terminal execution rewrites the file on disk).
```bash
git add deliverable_3/d3.ipynb
git commit -m "chore(d3): final clean-kernel execution"
```

---

## Self-review notes (author check against the spec)

- **Spec coverage:** Section 1 (Task 12) ↔ motivation; Section 2 (Task 2) ↔ data+quality; Section 3 (Tasks 3-5) ↔ preprocessing+features incl. all four new spatial features; Section 4 (Task 6) ↔ EDA; Section 5 (Task 7) ↔ methods; Section 6 (Tasks 7-10) ↔ validation incl. cross-city transfer + learning curves; Section 7 (Task 11) ↔ bootstrapped uncertainty + limitations; Section 8 (Task 12) ↔ decision; Section 9 (Task 12) ↔ conclusion. The four new analyses map to Tasks 4 (features), 9 (transfer), 11 (uncertainty), 10 (learning curves).
- **Leakage:** kNN-neighbour p95 is the one temperature-derived feature; it is injected only through `KNNNeighbourP95` (fit on train rows) inside the pipeline, and `p95` is excluded from `NUM_FEATURES`. Verified by the smoke test (Task 5) and the cross-validated contrast (Task 7).
- **Type consistency:** `make_pipeline(classifier, add_knn=False)` signature is fixed in Task 5 and used unchanged in Tasks 7, 9, 10. `proba` keys are 3-tuples `(target, model, featureset)` set in Task 7 and read with the same shape in Task 11. `rf.best_params_` keys are stripped of the `clf__` prefix consistently wherever a fresh RF is built.
- **No placeholders at submission:** `fill after running` appears only in intermediate tasks and is asserted to be 0 in Tasks 12 and 13.
