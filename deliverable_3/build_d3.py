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

# ---- Section 1 (stub; full prose filled in Task 12) -------------------
md(r"""
## 1. Question and motivation

`fill after running` — elaborate motivation written in Task 12.
""")

# ---- Section 2 --------------------------------------------------------
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

code(r'''
ARCHIVE = Path('../archive')
HOURLY_DIR = ARCHIVE / 'data' / 'hourly'
META_DIR   = ARCHIVE / 'data' / 'metadata'
CACHE_DIR  = Path('cache'); CACHE_DIR.mkdir(exist_ok=True)
HOURLY_PARQUET = CACHE_DIR / 'hourly_long.parquet'

assert HOURLY_DIR.exists(), f'Dataset not found at {HOURLY_DIR.resolve()}'

def parse_sef_file(path: Path) -> pd.DataFrame:
    """Parse one FAIRUrbTemp SEF TSV into long format (city, station_id, datetime, ta).

    station_id is the filename stem before '_AirTemp_' (handles 'D2195' and
    'AWEL_Blumenfeld'), Unicode-normalized to NFC so umlaut station names match
    the metadata TSVs. '-999' is a missing-value sentinel folded into NA.
    """
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
''')

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
