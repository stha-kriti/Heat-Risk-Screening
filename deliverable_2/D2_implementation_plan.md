# D2 — Local-Hotspot Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `deliverable_2/d2.ipynb` — a reproducible notebook that classifies each weather station as a within-city heat-risk "hotspot" (top-quartile summer p95 `ta` inside its own city), comparing a baseline to an improved model with rigorous cross-validation and a decision-threshold analysis.

**Architecture:** Single Jupyter notebook, six sections mapped 1:1 to the D2 rubric (see `deliverable_2/D2_design.md`). Self-contained: own copy of the SEF-parsing + risk-metric code, reads the shared dataset at `../archive/`, caches a parquet in `deliverable_2/cache/`. Models are sklearn `Pipeline`s (impute → scale → classifier) so preprocessing fits per-fold inside cross-validation (no leakage). Verification is per-cell: run, check printed output / inline `assert`s; final integration test runs the whole notebook headlessly.

**Tech Stack:** Python 3.12, pandas, numpy, matplotlib, seaborn, scikit-learn (all pinned in repo-root `requirements.txt`); `.venv` + Jupyter kernel `cdap` already registered.

**Notebook style:** cross-reference other sections as "Section 3.1" — never the `§` symbol. Tag each figure with its purpose where natural. Match the D1 markdown style.

---

## File structure

- **Create:** `deliverable_2/d2.ipynb` — the entire deliverable.
- **Create (runtime):** `deliverable_2/cache/hourly_long.parquet` — git-ignored parquet cache, regenerated from `../archive/` if missing.
- **Read-only:** `../archive/data/hourly/<City>/*.tsv` and `../archive/data/metadata/<City>/station_metadata_*.tsv`.
- **Reference:** `deliverable_2/D2_design.md` (the approved spec), `deliverable_1/d1.ipynb` (source of the SEF parser + risk-metric logic to copy).

A single notebook is the correct unit here — the deliverable IS one notebook, and the sections are tightly coupled (each consumes the prior section's in-memory objects).

---

## Build mechanics (how to edit a notebook from this plan)

The notebook is a JSON file. The reliable way to author it is a small Python script that assembles a list of cells and writes valid `.ipynb` JSON (the same method used for D1). For each task below, *append* the specified markdown + code cells to the notebook in order. After all cells are added, execute headlessly:

```bash
cd deliverable_2
../.venv/bin/jupyter nbconvert --to notebook --execute d2.ipynb --inplace \
  --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=cdap
```

Expected: `[NbConvertApp] Writing ... bytes to d2.ipynb` with no error traceback.

Helper for cell construction (use in the build script):

```python
import json
from pathlib import Path

cells = []
def md(text):   cells.append({'cell_type':'markdown','metadata':{},'source':text})
def code(text): cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':text})
# ... append cells per task ...
nb = {'cells':cells,'metadata':{'kernelspec':{'display_name':'Python (CDAP)','language':'python','name':'cdap'},
      'language_info':{'name':'python','version':'3.12'}},'nbformat':4,'nbformat_minor':5}
Path('deliverable_2/d2.ipynb').write_text(json.dumps(nb, indent=1, ensure_ascii=False))
```

(`source` can be a plain string; nbformat accepts it. Keep `ensure_ascii=False` so °C and arrows render.)

---

## Task 1: Notebook scaffold + Section 0 (setup) and Section 1 (recap)

**Files:** Create `deliverable_2/d2.ipynb`

- [ ] **Step 1: Title + Section 1 markdown cell**

```markdown
# Heat-Risk Screening — Deliverable 2

**Course:** Computational Data Analysis with Python (CDAP), TUM
**Track:** Track 1 — Heat-risk screening
**Group:** Jonas Juchelka & Kriti Shrestha
**Date:** June 2026

**Dataset:** FAIRUrbTemp curated subset (Amini et al., 2026, *Scientific Data*) — shared at `../archive/`

---

## 1. From D1 to D2

D1 ranked stations by absolute heat exposure and found the ranking was dominated by one city (Zurich), because climate (between cities) swamps microclimate (within city). D1's open question was to isolate microclimate using a within-city view.

D2 answers that as a **supervised classification** problem: for each station, predict whether it is a **local hotspot** — in the top quartile of summer p95 air temperature *within its own city* — from its location and sensor context. The decision threshold of the classifier is the screening decision itself: *how aggressively do we flag stations for mitigation investigation?*

**Section map (mapped to the D2 rubric):**

| Section | Rubric category | Weight |
|---|---|---:|
| 3. Preprocessing pipeline | Data preprocessing | 25% |
| 4. Models (baseline → improved) | Method selection & implementation | 25% |
| 5. Validation, metrics, threshold | Validation & evaluation | 30% |
| 6. Interpretation & decision | Interpretation | 10% |
| Structure & reproducibility | Communication | 10% |
```

- [ ] **Step 2: Section 0 markdown cell**

```markdown
## 0. Setup & imports
```

- [ ] **Step 3: Setup code cell**

```python
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    precision_score, recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, ConfusionMatrixDisplay,
)

RANDOM_STATE = 42
sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams['figure.figsize'] = (10, 5)

print('Libraries loaded.')
print(f'pandas {pd.__version__}, numpy {np.__version__}, sklearn loaded')
```

- [ ] **Step 4: Build the notebook file with these three cells and run it**

Run:
```bash
cd deliverable_2 && ../.venv/bin/jupyter nbconvert --to notebook --execute d2.ipynb --inplace --ExecutePreprocessor.timeout=120 --ExecutePreprocessor.kernel_name=cdap
```
Expected: no traceback; the setup cell prints "Libraries loaded." and the versions line.

---

## Task 2: Section 2 — Data loading & target construction

**Files:** Modify `deliverable_2/d2.ipynb` (append cells)

- [ ] **Step 1: Section 2 intro markdown cell**

```markdown
---

## 2. Data loading & target construction

This section is self-contained — it re-parses the FAIRUrbTemp SEF files (own copy of D1's loader), recomputes the per-station summer heat-risk metric, and builds the binary target. The dataset is read from the shared `../archive/`; a parquet cache is written to `cache/` so re-runs are fast.

**Target:** `is_local_hotspot` = 1 if a station's summer p95 `ta` is in the top quartile **within its own city**, else 0. The within-city definition isolates microclimate from climate (D1's open question) and keeps the classes balanced across cities.
```

- [ ] **Step 2: SEF loader code cell**

```python
ARCHIVE = Path('../archive')
HOURLY_DIR = ARCHIVE / 'data' / 'hourly'
META_DIR   = ARCHIVE / 'data' / 'metadata'
CACHE_DIR  = Path('cache'); CACHE_DIR.mkdir(exist_ok=True)
HOURLY_PARQUET = CACHE_DIR / 'hourly_long.parquet'

assert HOURLY_DIR.exists(), f'Dataset not found at {HOURLY_DIR.resolve()}'

def parse_sef_file(path: Path) -> pd.DataFrame:
    """Parse one FAIRUrbTemp SEF TSV into long format (city, station_id, datetime, ta).

    station_id is the filename stem before '_AirTemp_' (handles 'D2195' and 'AWEL_Blumenfeld').
    '-999' is a missing-value sentinel folded into NA alongside the literal 'NA'.
    """
    df = pd.read_csv(
        path, sep='\t', skiprows=12, na_values=['NA', '-999', '-999.0'],
        dtype={'Year':'Int64','Month':'Int64','Day':'Int64','Hour':'Int64','Minute':'Int64'},
    )
    parts = df[['Year','Month','Day','Hour','Minute']].rename(columns=str.lower)
    return pd.DataFrame({
        'city': path.parent.name,
        'station_id': path.stem.split('_AirTemp_')[0],
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

print(f'Loaded {len(hourly):,} rows, {hourly.station_id.nunique()} stations')
```

Expected: prints ~16.9M rows, 712 stations.

- [ ] **Step 3: Risk-metric + label code cell**

```python
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

# within-city top quartile -> binary hotspot label
risk['is_local_hotspot'] = (
    risk.groupby('city', observed=True)['p95']
        .transform(lambda s: (s >= s.quantile(0.75)).astype(int))
)

print(f'Stations after availability filter: {len(risk)}')
print(f'Cities: {risk.city.nunique()}')
print(f'Hotspot rate: {risk.is_local_hotspot.mean():.0%}')
print(risk.groupby('city', observed=True)['is_local_hotspot'].agg(['size','sum']))

assert 350 <= len(risk) <= 400, f'unexpected station count {len(risk)}'
assert 0.20 <= risk.is_local_hotspot.mean() <= 0.30, 'hotspot rate off'
```

Expected: ~375 stations, 6 cities, hotspot rate ~26%. Asserts pass.

- [ ] **Step 4: Metadata merge + feature engineering code cell**

```python
station_meta = pd.concat(
    [pd.read_csv(p, sep='\t') for p in sorted(META_DIR.glob('*/station_metadata_*.tsv'))],
    ignore_index=True,
)

def parse_height(v):
    """'4 m' -> 4.0 ; '2-3 m' -> 2.5 ; 2.0 -> 2.0 ; junk -> NaN"""
    if pd.isna(v): return np.nan
    s = str(v).lower().replace('m', '').strip()
    if '-' in s:
        a, b = s.split('-', 1)
        try: return (float(a) + float(b)) / 2
        except ValueError: return np.nan
    try: return float(s)
    except ValueError: return np.nan

station_meta['sensor_height_parsed'] = station_meta['sensor_height_m'].apply(parse_height)

df = risk.merge(
    station_meta[['station_id', 'latitude', 'longitude', 'sensor_height_parsed']],
    on='station_id', how='left',
)

# City centroid over the modeling stations (pure geography, target-independent -> no target leakage)
cent = (df.groupby('city', observed=True)[['latitude', 'longitude']]
          .mean().rename(columns={'latitude': 'clat', 'longitude': 'clon'}))
df = df.merge(cent, on='city')

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1); dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat/2)**2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
    return 2 * R * np.arcsin(np.sqrt(a))

df['distance_to_centroid'] = haversine_km(df.latitude, df.longitude, df.clat, df.clon)

FEATURES = ['latitude', 'longitude', 'distance_to_centroid', 'sensor_height_parsed']
TARGET = 'is_local_hotspot'

print(df[FEATURES + [TARGET]].describe(include='all').round(2))
print('\nMissing per feature:')
print(df[FEATURES].isna().sum())
assert df[TARGET].notna().all(), 'missing labels'
```

Expected: a describe table; `sensor_height_parsed` may have some NaN (imputed later); other features fully populated; asserts pass.

- [ ] **Step 5: Interpretation markdown cell**

```markdown
**Notes on features (preprocessing rationale, see Section 3):**
- `latitude`, `longitude`, `distance_to_centroid` are pure geography — independent of the temperature record that defines the target, so they cannot leak the label.
- `distance_to_centroid` is the within-city-position signal (centre → denser → warmer, the urban-heat-island effect). Because it is target-independent, computing each city centroid once over all modeling stations introduces no target leakage.
- `sensor_height_m` is parsed from free text and has gaps; it is imputed inside the modeling pipeline (Section 3) so the fill value is learned per training fold.
- `altitude_m` is excluded — populated only for Zurich/Berlin/NoviSad, unusable across the 6 modeling cities.
- `city` is excluded as a predictor — the target is already defined within-city, so re-adding city would be circular.
```

- [ ] **Step 6: Run notebook, confirm Section 2 executes**

Run the nbconvert command from Task 1 Step 4 (timeout 600). Expected: no traceback; Section 2 prints the station count (~375), hotspot rate (~26%), and feature describe table.

---

## Task 3: Section 3 — Preprocessing pipeline definition

**Files:** Modify `deliverable_2/d2.ipynb` (append cells)

- [ ] **Step 1: Section 3 markdown cell**

```markdown
---

## 3. Preprocessing pipeline

All models share the same preprocessing, wrapped in an sklearn `Pipeline` so that the imputer and scaler are fit on **training folds only** during cross-validation — preventing leakage. Tree models do not need scaling, but a `StandardScaler` is harmless for them and keeps one shared preprocessing definition.

Steps: median-impute `sensor_height_parsed` (the only feature with gaps) → standardize all four features → classifier.
```

- [ ] **Step 2: Pipeline factory code cell**

```python
X = df[FEATURES].copy()
y = df[TARGET].copy()

def make_pipeline(classifier):
    """Shared preprocessing (impute -> scale) + a classifier, leak-free under CV."""
    return Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('clf', classifier),
    ])

# sanity: pipeline fits and predicts on the full data (smoke test only, not evaluation)
_smoke = make_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
_smoke.fit(X, y)
print('Pipeline smoke test OK; predicts shape:', _smoke.predict(X).shape)
assert _smoke.predict_proba(X).shape == (len(X), 2)
```

Expected: "Pipeline smoke test OK; predicts shape: (375,)"; assert passes.

- [ ] **Step 3: Run notebook through Section 3**

Run nbconvert. Expected: no traceback; smoke test line prints.

---

## Task 4: Section 4 — Models (baseline → improved)

**Files:** Modify `deliverable_2/d2.ipynb` (append cells)

- [ ] **Step 1: Section 4 markdown cell**

```markdown
---

## 4. Models: baseline → improved

We escalate three models, all on the same features and preprocessing:

1. **Baseline 1 — `DummyClassifier(stratified)`** — the no-skill floor; any useful model must beat it.
2. **Baseline 2 — `LogisticRegression(class_weight='balanced')`** — interpretable; its coefficients show which geographic factors push a station toward hotspot.
3. **Improved — `RandomForestClassifier(class_weight='balanced')`** — captures non-linear geography.

`class_weight='balanced'` handles the 26%/74% imbalance without resampling (SMOTE is avoided — N is small and synthetic points would add noise). Hyperparameters are tuned with a small `GridSearchCV` inside cross-validation.
```

- [ ] **Step 2: Define the three estimators code cell**

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

dummy = make_pipeline(DummyClassifier(strategy='stratified', random_state=RANDOM_STATE))

logit = GridSearchCV(
    make_pipeline(LogisticRegression(class_weight='balanced', max_iter=1000,
                                     random_state=RANDOM_STATE)),
    param_grid={'clf__C': [0.1, 1.0, 10.0]},
    scoring='average_precision', cv=cv, n_jobs=-1,
)

rf = GridSearchCV(
    make_pipeline(RandomForestClassifier(class_weight='balanced',
                                         random_state=RANDOM_STATE, n_jobs=-1)),
    param_grid={'clf__n_estimators': [200, 400], 'clf__max_depth': [3, 5, None]},
    scoring='average_precision', cv=cv, n_jobs=-1,
)

logit.fit(X, y)
rf.fit(X, y)
print('LogReg best params:', logit.best_params_)
print('RF best params:    ', rf.best_params_)
```

Expected: prints best params for each (e.g. `clf__C: 1.0`; some RF depth/estimators combo). No error.

- [ ] **Step 3: Run notebook through Section 4**

Run nbconvert (timeout 600). Expected: no traceback; both best-params lines print.

---

## Task 5: Section 5 — Validation, metrics, threshold tuning, error analysis

**Files:** Modify `deliverable_2/d2.ipynb` (append cells)

- [ ] **Step 1: Section 5 markdown cell**

```markdown
---

## 5. Validation & evaluation

We use **5-fold stratified cross-validation** (preserving the 26% hotspot rate per fold) and evaluate on **out-of-fold** predictions, so every metric reflects performance on stations the model did not train on. We emphasise **PR-AUC (average precision)** because hotspots are the minority class.

Cross-city transfer (training on some cities, testing on a held-out city) is a stronger robustness test deferred to D3.
```

- [ ] **Step 2: Out-of-fold probabilities + metrics table code cell**

```python
def oof_proba(estimator):
    """Out-of-fold positive-class probabilities via stratified CV."""
    return cross_val_predict(estimator, X, y, cv=cv, method='predict_proba')[:, 1]

models = {
    'Dummy (stratified)': dummy,
    'LogReg (balanced)':  logit.best_estimator_,
    'RandomForest':       rf.best_estimator_,
}
proba = {name: oof_proba(est) for name, est in models.items()}

def metrics_at(p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return {
        'ROC-AUC': roc_auc_score(y, p),
        'PR-AUC':  average_precision_score(y, p),
        'Precision': precision_score(y, yhat, zero_division=0),
        'Recall':    recall_score(y, yhat, zero_division=0),
        'F1':        f1_score(y, yhat, zero_division=0),
        'MCC':       matthews_corrcoef(y, yhat),
    }

results = pd.DataFrame({name: metrics_at(p) for name, p in proba.items()}).T.round(3)
print('Out-of-fold metrics at threshold 0.5:')
print(results)
assert results.loc['LogReg (balanced)', 'ROC-AUC'] > results.loc['Dummy (stratified)', 'ROC-AUC'], \
    'LogReg should beat the dummy baseline'
```

Expected: a metrics table; LogReg and RF ROC-AUC noticeably above the dummy (~0.5). Assert passes. (Honest expectation: real-model ROC-AUC ~0.65–0.80.)

- [ ] **Step 3: ROC + PR curve plot code cell**

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for name, p in proba.items():
    fpr, tpr, _ = roc_curve(y, p)
    axes[0].plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(y, p):.2f})')
    prec, rec, _ = precision_recall_curve(y, p)
    axes[1].plot(rec, prec, label=f'{name} (AP={average_precision_score(y, p):.2f})')
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.4)
axes[0].set(title='ROC curve (out-of-fold)', xlabel='False positive rate', ylabel='True positive rate')
axes[1].axhline(y.mean(), ls='--', color='k', alpha=0.4, label=f'baseline rate={y.mean():.2f}')
axes[1].set(title='Precision-Recall curve (out-of-fold)', xlabel='Recall', ylabel='Precision')
axes[0].legend(); axes[1].legend()
plt.tight_layout(); plt.show()
```

Expected: two-panel figure; real models bow above the diagonal / baseline.

- [ ] **Step 4: Threshold tuning (the screening decision) code cell**

```markdown
**Threshold = the screening decision.** A lower threshold flags more stations (catches more true hotspots = higher recall, but more false alarms = lower precision). We make the trade-off explicit with a cost ratio: missing a real hotspot is treated as `COST_FN` times worse than a false alarm.
```

```python
best_name = 'RandomForest' if results.loc['RandomForest','PR-AUC'] >= results.loc['LogReg (balanced)','PR-AUC'] else 'LogReg (balanced)'
p_best = proba[best_name]
COST_FN, COST_FP = 3, 1   # missing a hotspot is 3x worse than a false alarm

thresholds = np.linspace(0.05, 0.95, 91)
costs, precs, recs = [], [], []
for t in thresholds:
    yhat = (p_best >= t).astype(int)
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    costs.append(COST_FP*fp + COST_FN*fn)
    precs.append(precision_score(y, yhat, zero_division=0))
    recs.append(recall_score(y, yhat, zero_division=0))
t_star = float(thresholds[int(np.argmin(costs))])

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(thresholds, precs, label='Precision')
ax.plot(thresholds, recs, label='Recall')
ax.axvline(t_star, color='red', ls='--', label=f'cost-optimal threshold = {t_star:.2f}')
ax.set(title=f'Precision/Recall vs threshold — {best_name} (cost FN:FP = {COST_FN}:{COST_FP})',
       xlabel='Decision threshold', ylabel='Score')
ax.legend(); plt.tight_layout(); plt.show()

print(f'Chosen model: {best_name}')
print(f'Cost-optimal threshold: {t_star:.2f}')
print('Metrics at chosen threshold:', {k: round(v,3) for k,v in metrics_at(p_best, t_star).items()})
```

Expected: a precision/recall-vs-threshold figure with the cost-optimal threshold marked; printed threshold and metrics.

- [ ] **Step 5: Confusion matrix + error analysis code cell**

```python
yhat_star = (p_best >= t_star).astype(int)
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(confusion_matrix(y, yhat_star),
                       display_labels=['Not hotspot', 'Hotspot']).plot(ax=ax, colorbar=False)
ax.set_title(f'{best_name} @ threshold {t_star:.2f}')
plt.tight_layout(); plt.show()

err = df.assign(y_true=y.values, y_pred=yhat_star, p_hot=p_best)
misclassified = err[err.y_true != err.y_pred][
    ['city', 'station_id', 'p95', 'distance_to_centroid', 'sensor_height_parsed', 'y_true', 'p_hot']
].sort_values('p_hot')
print(f'Misclassified stations: {len(misclassified)} of {len(err)}')
print('By city:'); print(misclassified.city.value_counts())
print(misclassified.head(10).to_string(index=False))
```

Expected: confusion matrix figure; a misclassified-stations table with city breakdown.

- [ ] **Step 6: Interpretation markdown cell (fill after running)**

```markdown
**Interpretation (fill after running):**

- *Best model and its out-of-fold ROC-AUC / PR-AUC: …*
- *Does it clearly beat the dummy and the logistic baseline? By how much?*
- *At the cost-optimal threshold, precision = … and recall = … — in screening terms, we catch …% of true local hotspots while …% of flagged stations are false alarms.*
- *Which stations are misclassified, and do they share traits (edge-of-city distance, unusual sensor height, a particular city)?*
```

- [ ] **Step 7: Run notebook through Section 5**

Run nbconvert (timeout 600). Expected: no traceback; metrics table, ROC/PR figure, threshold figure, confusion matrix, misclassified table all render.

---

## Task 6: Section 6 — Interpretation, decision, limitations, D3 hand-off

**Files:** Modify `deliverable_2/d2.ipynb` (append cells)

- [ ] **Step 1: Section 6 markdown cell**

```markdown
---

## 6. Interpretation, decision & limitations
```

- [ ] **Step 2: Feature-importance code cell**

```python
# Logistic coefficients (standardized features -> directly comparable)
logit_clf = logit.best_estimator_.named_steps['clf']
coef = pd.Series(logit_clf.coef_[0], index=FEATURES).sort_values()

# RF importances
rf_clf = rf.best_estimator_.named_steps['clf']
imp = pd.Series(rf_clf.feature_importances_, index=FEATURES).sort_values()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
coef.plot.barh(ax=axes[0], color='#4C72B0')
axes[0].axvline(0, color='k', lw=0.8)
axes[0].set_title('Logistic coefficients (standardized)')
imp.plot.barh(ax=axes[1], color='#C44E52')
axes[1].set_title('Random Forest feature importances')
plt.tight_layout(); plt.show()

print('Logistic coefficients:'); print(coef.round(3))
print('\nRF importances:'); print(imp.round(3))
```

Expected: two-panel feature-importance figure; printed coefficient and importance values.

- [ ] **Step 3: Decision + limitations markdown cell (fill after running)**

```markdown
### 6.1 What predicts a local hotspot (fill after running)

- *Sign and size of the `distance_to_centroid` coefficient — does being closer to the city centre raise hotspot probability, as the urban-heat-island hypothesis predicts? …*
- *Do logistic coefficients and RF importances agree on the dominant feature? …*

### 6.2 Screening recommendation

At the cost-optimal threshold (Section 5), the model flags stations scoring above it for site investigation. For a planner with a fixed budget, this gives a within-city shortlist that catches *[recall]%* of true local hotspots at *[precision]%* precision. The threshold can be moved per the planner's tolerance for missed hotspots vs wasted visits.

### 6.3 Limitations

- **Thin features.** Only geography + sensor height. The true drivers of microclimate (land cover, imperviousness, building density, proximity to water/green) are not in this dataset; modest performance is expected and honest.
- **Relative label.** A "hotspot" is top-quartile *within its city* — a hotspot in cool Amsterdam is not hot in absolute terms. Read alongside D1's absolute view.
- **Small N** (~375 stations across 6 cities); within-city quartiles for the smaller cities (Amsterdam, Berlin: 10 stations each) rest on very few stations.
- **No causal claim** — geography is associated with, not a cause of, measured heat.

### 6.4 Hand-off to D3

- Integrate the absolute (D1) and relative (D2) views into one screening recommendation.
- Run the deferred **cross-city transfer** test (GroupKFold by city): does the within-city geographic pattern generalize to a city never seen in training?
- Identify the external data (land cover, satellite LST) that would most improve discrimination.
```

- [ ] **Step 4: Final headless execution (integration test)**

Run:
```bash
cd deliverable_2 && ../.venv/bin/jupyter nbconvert --to notebook --execute d2.ipynb --inplace --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=cdap
```
Expected: `[NbConvertApp] Writing ... bytes to d2.ipynb`, zero error outputs.

- [ ] **Step 5: Verify no errors and no `§` symbol**

Run:
```bash
cd deliverable_2 && python3 -c "
import json,re
nb=json.load(open('d2.ipynb'))
errs=sum(1 for c in nb['cells'] for o in c.get('outputs',[]) if o.get('output_type')=='error')
txt='\n'.join(''.join(c['source']) for c in nb['cells'])
print('cells:',len(nb['cells']),'errors:',errs,'section-symbols:',txt.count(chr(167)))
assert errs==0 and chr(167) not in txt
print('OK')
"
```
Expected: `errors: 0`, `section-symbols: 0`, prints `OK`.

---

## Verification (end-to-end)

1. From a clean kernel, the notebook runs top-to-bottom with no errors (Task 6 Step 4).
2. Section 2 produces ~375 stations, 6 cities, ~26% hotspot rate (asserts in Task 2).
3. Both real models beat the dummy baseline on ROC-AUC (assert in Task 5 Step 2).
4. All figures render: ROC/PR curves, threshold trade-off, confusion matrix, feature importances.
5. No `§` symbol anywhere (Task 6 Step 5).
6. The four interpretation markdown cells (Section 5 Step 6, Section 6 Steps 1/3) remain as "fill after running" prompts for Kriti + Jonas to complete with the real numbers — mirroring the D1 workflow.

## Out of scope (per spec)

- Cross-city GroupKFold transfer validation (→ D3).
- Hourly/event-level prediction (Track 2).
- A second parallel modeling task (the rubric rewards depth on one task).
- Acquiring external features (noted as the key limitation).
