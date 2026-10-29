# D2 Design — Heat-Risk Screening: Local-Hotspot Classification

**Course:** CDAP (TUM) · **Track 1 — Heat-risk screening** · Group: Jonas Juchelka & Kriti Shrestha
**Deliverable:** D2 (Data analysis & prediction), due **June 7, 2026**. Weight: 15%.
**Status:** design approved 2026-05-31; spec for review before implementation planning.

---

## Context — why this design

D1 framed the screening question ("which locations experience the highest heat exposure?") and produced per-station heat-risk metrics (summer p95 `ta`, 30 °C exceedance rate) plus a data-quality audit. Its headline finding: an **absolute** risk ranking is dominated by one city (Zurich), because climate (between cities) swamps microclimate (within city). D1 explicitly flagged "use within-city z-scored anomalies to isolate microclimate" as the open question for D2.

D2 turns that screening question into a **supervised model with rigorous validation**, as the D2 rubric demands (Method 25% + Validation 30% = 55% of the grade) and as the newly-released course notebooks (S5 regression, S6 classification, T5 pipeline) signal is expected. We keep the **station as the unit of analysis** (continuity with D1) and choose a **classification** framing because (a) it unlocks the full S6 validation suite — ROC, PR, confusion matrix, threshold tuning — and (b) the decision threshold *is* the screening decision ("how aggressively do we flag stations for intervention?").

The intended outcome: a defensible, reproducible model that flags within-city heat-risk hotspots from location/context, validated honestly, with the decision threshold tied to a mitigation-prioritisation trade-off.

## Rubric map (D2)

| Section | Rubric category | Weight |
|---|---|---:|
| Section 3 — Preprocessing pipeline | Data preprocessing | 25% |
| Section 4 — Models (baseline → improved) | Method selection & implementation | 25% |
| Section 5 — Validation, metrics, threshold, error analysis | Validation & evaluation | 30% |
| Section 6 — Interpretation & decision | Interpretation of results | 10% |
| Notebook structure & reproducibility | Communication & reproducibility | 10% |

---

## 1. Task & target

- **Task:** binary classification of `is_local_hotspot`.
- **Target definition:** a station is a *local hotspot* if its summer p95 `ta` is in the **top quartile within its own city**.
  - **Why within-city, not global:** a global top-25% label puts 76% of positives in Zurich (72 of 95) — a classifier given `city` would trivially "predict Zurich" and learn nothing. The within-city label spreads positives across all cities (Zurich 46, Basel 25, Bern 14, Birmingham 5, Amsterdam 3, Berlin 3) and isolates **microclimate from climate** — directly delivering D1's open question. It also matches the real decision: a planner asks "within *my* city and budget, which stations are the local hotspots?"
  - **Caveat to state plainly:** the label is *relative*. A "hotspot" in cool Amsterdam is not hot in absolute terms. The decision-maker must read D2 (relative, within-city) alongside D1 (absolute) — this is the D3 integration point.
- **Unit of analysis:** station.
- **Population:** the **375 stations** passing D1's ≥50% summer-availability filter, across **6 cities** (Zurich 184, Basel 97, Bern 55, Birmingham 19, Amsterdam 10, Berlin 10). Novi Sad and Biel are excluded — outside the 2019–2022 summer window (per D1).
- **Class balance:** ~26% positive (96/375). No resampling strictly required; we use `class_weight='balanced'` and discuss the choice (S6 material). SMOTE is explicitly *not* used (small N, would add noise).

## 2. Data loading

- `deliverable_2/` gets **its own copy** of the SEF-parsing + risk-metric code (self-contained per-deliverable convention; no cross-folder imports).
- Reads the dataset from the shared repo-root `../archive/` (git-ignored; both deliverables read from this one neutral location).
- Writes its own parquet cache to `deliverable_2/cache/` (git-ignored), regenerated from `../archive/` if missing.
- The risk-metric computation (summer 2019–2022 filter, ≥50% availability, per-station p95) is reproduced verbatim from D1 so D2 stands alone.

## 3. Features & preprocessing pipeline — *Preprocessing 25%*

**Features used (all pure geography/metadata — independent of the temperature record, so not circular):**

| Feature | Source | Notes |
|---|---|---|
| `latitude`, `longitude` | station_metadata | clean, available for all stations |
| `distance_to_city_centroid` | engineered | great-circle distance from station to its city's centroid; the within-city geography signal (centre = denser = hotter, urban heat island). City centroid computed from **train stations only** to avoid leakage. |
| `sensor_height_m` | station_metadata | free-text (`'4 m'`, `2.0`, `'2-3 m'`) → parsed to float; nearly constant (2–4 m) → weak but documented |

**Deliberately excluded, with documented reasons:**
- `altitude_m` — populated only for Zurich, Berlin, Novi Sad; missing for Basel, Bern, Amsterdam, Birmingham → unusable as a cross-city feature.
- `city` as a predictor — the target is already defined within-city, so feeding city back in is circular. `city` is retained **only as the grouping key** for the deferred D3 cross-city analysis and for stratified reporting.

**Pipeline (sklearn `Pipeline` + `ColumnTransformer`, T5-style, leak-free):**
- parse `sensor_height_m` free-text → numeric (a small parsing function applied to the raw frame; deterministic, no fitted state).
- `SimpleImputer(strategy='median')` for parsed sensor height (has gaps).
- `StandardScaler` on all numeric features.
- All fitted state (imputer median, scaler mean/scale, city centroids for distance) is learned on **train folds only**; the whole transform is wrapped in the `Pipeline` so cross-validation calls `fit` only on training folds and `transform` on held-out folds. No leakage.

## 4. Models: baseline → improved — *Method 25%*

Following the course's "compare to a simple baseline" expectation (T5), one task with escalating models:

- **Baseline 1 — `DummyClassifier(strategy='stratified')`:** the no-skill floor; any real model must beat it.
- **Baseline 2 — `LogisticRegression(class_weight='balanced', max_iter=1000)`:** interpretable primary model (S6). Coefficients reveal which geographic factors push a station toward hotspot.
- **Improved — `RandomForestClassifier(class_weight='balanced', random_state=42)`:** captures non-linear geography (hotspots may cluster centrally *and* in specific pockets). Compared head-to-head with logistic.
- **Hyperparameters:** small `GridSearchCV` grid *inside* the CV (T5 pattern) — e.g. LogReg `C`, RF `n_estimators`/`max_depth`. Kept small (novice-friendly, fast, no overfitting the grid).

## 5. Validation, metrics, threshold — *Validation 30% (the heavy section)*

- **Primary validation — `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`:** preserves the 26% positive rate per fold. Matches the screening use-case ("we have sensors in all 6 cities; find the hotspots among them"). Use `cross_val_predict(..., method='predict_proba')` for out-of-fold probabilities (S6 idiom).
- **Cross-city transfer (Group-by-city) — DEFERRED to D3.** One paragraph in D2 notes it as a planned robustness check: "can the within-city pattern transfer to a city never seen in training?" Out of D2 scope to keep the narrative focused and the 30% weight on the use-case that matters.
- **Metrics (full S6 suite):** ROC-AUC, PR-AUC / average precision (emphasised — positives are the minority), precision, recall, F1, MCC, balanced accuracy, confusion matrix.
- **Threshold tuning = the screening decision:** sweep the decision threshold; plot precision & recall vs threshold and the PR curve; frame as **cost of a missed hotspot** (under-protect a hot area) vs **cost of a false alarm** (waste mitigation budget). Select and justify an operating point. This is where Validation-30% meets Track-1's "where to prioritise" decision.
- **Error analysis:** inspect misclassified stations — do false negatives/positives share traits (edge-of-city distance, unusual sensor height, specific city)?

## 6. Interpretation, decision & limitations — *Interpretation 10%*

- **Drivers:** logistic coefficients + RF feature importances → *what geography predicts a local hotspot* (expectation: smaller distance-to-centroid ⇒ higher hotspot probability).
- **Decision:** the chosen threshold yields a concrete recommendation — "flag stations scoring above X for site investigation," with the precision/recall trade-off made explicit for the planner.
- **Honest performance read:** with only lat/lon/distance/sensor-height, expect **modest** discrimination (AUC ≈ 0.65–0.75). State it plainly; explain *why* (geography is a weak proxy for the true drivers — land cover, imperviousness, building density — which are not in this dataset). The rubric rewards honest evaluation over high accuracy.
- **Limitations:** thin features; climate excluded by design; small N; relative (within-city) label.
- **D3 hand-off:** combine D1's absolute view and D2's relative view into one integrated screening recommendation; run the deferred cross-city transfer test; discuss what external data would lift performance.

---

## Notebook structure (mirrors rubric; matches D1's tagged style)

0. Setup & imports (same venv/kernel `Python (CDAP)`; `scikit-learn` already pinned in `requirements.txt`)
1. Recap & D2 question (one paragraph bridging D1 → D2)
2. Data loading & target construction (self-contained; build `is_local_hotspot`)
3. Features & preprocessing pipeline
4. Models: baseline → improved
5. Validation, metrics, threshold tuning, error analysis
6. Interpretation, decision, limitations, D3 hand-off

## Layout & mechanics

- New flat folder `deliverable_2/` containing `d2.ipynb`, `D2_design.md` (this file), `cache/` (git-ignored).
- Reuses `.venv` and the `cdap` Jupyter kernel; no new dependencies.
- `.gitignore` already covers `deliverable_*/cache/` and `*.parquet`.

## Out of scope for D2 (explicit)

- Cross-city (Group-by-city) transfer validation → D3.
- Hourly / event-level prediction (that is Track 2's question) → not pursued.
- A second, parallel modeling *task* (e.g. regression as well) → the rubric rewards depth on one task, not breadth; baseline-vs-improved within the single classification task is the intended "two models."
- Acquiring external features (land cover, etc.) → noted as the key limitation and D3 discussion, not collected.
