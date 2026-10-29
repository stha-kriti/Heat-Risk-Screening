# D3 Design — Heat-Risk Screening: Final Integrated Notebook

**Course:** CDAP (TUM) · **Track 1 — Heat-risk screening** · Group: Jonas Juchelka & Kriti Shrestha
**Deliverable:** D3 (Final integrated notebook), due **June 23, 2026**. Weight: 30%.
**Status:** design drafted 2026-06-18, for review before implementation planning.

---

## 1. What D3 is

D3 is the final, standalone notebook that tells the whole project as one story: question → data → analysis → results → interpretation → decision. It consolidates D1 (problem framing, data-quality audit, EDA) and D2 (within-city classification, the contrast finding) into a single coherent narrative, then pushes the analysis further with the pieces D2 deferred.

D3 is **self-contained**. It has its own data-loading code, reads the dataset from the shared repo-root `../archive/`, and runs top-to-bottom from a clean kernel with no dependency on the D1 or D2 notebooks. A first-time reader who has never seen D1 or D2 can follow it end to end.

## 2. The narrative spine

The argument the notebook makes, in order:

1. **The question.** Which locations carry the highest summer heat exposure, and how should a city screen for them? Heat is a public-health problem; screening lets a planner prioritise limited mitigation budget.
2. **Two views of risk.** *Absolute* exposure (raw summer p95 air temperature across all cities) and *within-city* microclimate (the same metric, but ranked inside each city so climate is removed). These answer different decisions.
3. **The contrast finding (from D2).** From geography alone, absolute risk is predictable (RandomForest out-of-fold ROC-AUC ≈ 0.69) but within-city risk sits at chance (≈ 0.50). Heat risk is predictable *between* cities, not *within* them, from the geographic features available.
4. **D3's new push — can we rescue the within-city model?** Engineer spatial features from the coordinates and metadata already in hand (no new external data). Re-test both targets. A lift means microclimate is partly recoverable from spatial structure; no lift sharpens the conclusion that land cover is the genuinely missing ingredient. Either outcome is a real result.
5. **How far do we trust this?** A cross-city transfer test and bootstrapped uncertainty intervals establish whether the contrast is a real, stable gap or an artefact of a small sample.
6. **The decision.** A concrete two-tier screening recommendation for a planner, with the chosen operating threshold and what external data would most improve the screen.

## 3. New analysis beyond consolidation

Four genuinely new pieces, all reproducible and offline:

1. **Spatial feature engineering (Option B — no new data).** Features derived purely from existing station coordinates and metadata:
   - **Local station density** — number of other stations within a fixed radius (≈ 1 km) of each station; a proxy for built-up density.
   - **kNN-neighbour p95** — mean summer p95 of the *k* nearest *other* stations; tests spatial autocorrelation ("is this station in a hot neighbourhood?"). Computed **leak-free inside CV folds** (neighbours and their p95 taken from training stations only).
   - **Bearing from city centroid** — direction (sin/cos of bearing, or N–S and E–W offsets) relative to the city centre, not just distance.
   - **Elevation** — `altitude_m` where populated (Zurich, Berlin, NoviSad); used as an additional numeric feature, documented as city-limited.
   Both targets are re-run with the enriched feature set and compared to the D2 geography-only baseline.

2. **Cross-city transfer — `GroupKFold` by city** (deferred from D2). Train on a subset of cities, predict a held-out city never seen in training. Answers "does the learned pattern generalise to a new city?" — the real test of a screening tool meant to be deployed in new places.

3. **Uncertainty quantification (from S8).** Bootstrapped confidence intervals on ROC-AUC and PR-AUC for both models, so the absolute-vs-within-city contrast is reported *with error bars*. We state whether 0.69-vs-0.50 is a real gap at this sample size. Plain-language aleatoric (irreducible measurement and microclimate noise) vs epistemic (reducible with more or better data) framing.

4. **Learning curves (from T6 / S7).** Train/validation score vs training-set size, to diagnose bias vs variance: would *more stations* lift performance, or are we feature-limited? This is the evidence behind the "what would improve the screen" recommendation, rather than an unsupported assertion.

## 4. Data, target, population (carried from D1/D2, restated standalone)

- **Source.** FAIRUrbTemp curated subset, read from `../archive/`. Per-station hourly air temperature (`ta`, °C) in SEF (Station Exchange Format); one measured variable.
- **Unit of analysis.** Station.
- **Risk metric.** Summer (Jun–Aug, 2019–2022) p95 of hourly `ta` per station, after a ≥50% availability filter.
- **Population.** The 375 stations passing the availability filter across 6 cities (Zurich 184, Basel 97, Bern 55, Birmingham 19, Amsterdam 10, Berlin 10). Novi Sad and Biel fall outside the summer window and are excluded.
- **Targets (both used, for the contrast).**
  - `is_hotspot_absolute` — p95 in the global top quartile.
  - `is_local_hotspot` — p95 in the top quartile *within its own city* (microclimate, climate removed).
- **Class balance.** ~26% positive. `class_weight='balanced'`; no resampling (small N).

## 5. Features and preprocessing

| Feature | Source | Role |
|---|---|---|
| `latitude`, `longitude` | station_metadata | base geography |
| `distance_to_centroid` | engineered (haversine) | within-city position; centroid from train stations only |
| `sensor_height_parsed` | station_metadata | free-text → float; median-imputed |
| `station_density` | engineered (NEW) | count of stations within ~1 km |
| `knn_neighbour_p95` | engineered (NEW) | mean p95 of k nearest other stations; leak-free in CV |
| `bearing` (sin/cos or N–S/E–W) | engineered (NEW) | direction from centroid |
| `altitude_m` | station_metadata (NEW use) | elevation where populated; city-limited |

- **Excluded, documented:** `city` as a predictor (target is already within-city for one target; circular) — retained only as the `GroupKFold` grouping key and for stratified reporting.
- **Pipeline:** sklearn `Pipeline` + `ColumnTransformer`. Median impute → `StandardScaler` on numeric features; all fitted state (medians, scaler stats, centroids, kNN neighbours) learned on train folds only. Cross-validation calls `fit` on train and `transform` on held-out folds. No leakage.

## 6. Models and validation

- **Models (baseline → improved):** `DummyClassifier(strategy='stratified')` floor → `LogisticRegression(class_weight='balanced')` (interpretable) → `RandomForestClassifier(class_weight='balanced', random_state=42)` (non-linear). Small `GridSearchCV` inside CV for the main hyperparameters.
- **Primary validation:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`; out-of-fold probabilities via `cross_val_predict(method='predict_proba')`.
- **Cross-city validation:** `GroupKFold` grouped by `city` (NEW in D3).
- **Metrics:** ROC-AUC, PR-AUC / average precision (emphasised — minority positives), precision, recall, F1, MCC, balanced accuracy, confusion matrix. Each explained in words before the numbers.
- **Threshold tuning = the screening decision:** sweep the decision threshold; plot precision & recall vs threshold and the PR curve; frame as cost of a missed hotspot vs cost of a false alarm; select and justify an operating point.
- **Uncertainty:** bootstrapped CIs on ROC-AUC / PR-AUC (NEW).
- **Diagnostics:** learning curves (NEW); error analysis on misclassified stations.

## 7. Notebook structure (Question → Data → Analysis → Results → Interpretation → Decision)

The flow follows the course workflow implicitly; rubric category names are **not** echoed in the notebook. Every section opens with an elaborate plain-language explanation so a first-time reader can follow the reasoning, not just the code.

0. **Setup** — imports, paths, seed, kernel note.
1. **Question and motivation** — public-health stakes, the screening question, what "heat risk" means (p95), the two lenses (absolute vs within-city), success criterion (honest, decision-useful screen — not raw accuracy).
2. **The data** — FAIRUrbTemp, SEF format, the one variable and its units, station as unit, the six cities and counts; then the standalone quality assessment (`NA` / `-999` sentinels, availability filter, Unicode station-id pitfall), each with why it matters.
3. **From raw records to a modeling table** — reproducible preprocessing: parse SEF → summer p95 → both targets → feature engineering (original + NEW spatial features, each explained) → leak-free pipeline with an explicit fit-on-train-only note.
4. **Exploratory data analysis** — the two findings that motivate the contrast (Zurich dominates the absolute ranking; real within-city microclimate spread); feature distributions and spatial scatter; each figure with a one-line "what this tells us".
5. **Modeling approach** — why classification suits a screening decision; baseline → improved ladder; both targets side by side; readable bias-variance intuition for expecting modest skill.
6. **Validation and results** — StratifiedKFold metrics for both targets (the contrast, now possibly moved by the new features); threshold tuning as the screening trade-off; cross-city `GroupKFold` transfer; learning curves; error analysis. Every metric explained before the numbers.
7. **Uncertainty and limitations** — bootstrapped CIs on the contrast; aleatoric vs epistemic in plain language; honest limitations (thin features, relative label, small N, climate excluded by design).
8. **Decision-oriented interpretation and recommendation** — the two-tier screening recommendation (absolute view to triage cities, within-city view to find local hotspots); chosen operating threshold and its practical meaning; what external data (real land cover / imperviousness) would most lift the screen, earned by the learning-curve evidence.
9. **Conclusion** — the arc in a few sentences: question → two views → the contrast → how far it's trustworthy → what to do.

## 8. Layout and mechanics

- New flat folder `deliverable_3/` containing `d3.ipynb`, `D3_design.md` (this file), `cache/` (git-ignored). Notebook directly at the top of the folder.
- Reads the dataset from the shared repo-root `../archive/`; writes its own parquet cache to `deliverable_3/cache/`, regenerated if missing. The dataset is **not** duplicated into the folder.
- Reuses `.venv` and the `cdap` Jupyter kernel; no new dependencies (sklearn, numpy, pandas, matplotlib, seaborn already pinned in `requirements.txt`).
- `.gitignore` already covers `deliverable_*/cache/` and `*.parquet`.
- Notebook style: cross-references written as "Section 3.1" (no `§` symbol).

## 9. Out of scope for D3 (explicit)

- **External data acquisition** (real land cover, imperviousness, population density) — named as the key limitation and the highest-value future work, **not** collected. Reproducibility and the course's "one shared dataset" principle rule out live web fetches; synthesising feature values is fabrication and is excluded outright.
- **Hourly / event-level prediction** (Track 2's question) — not pursued.
- **A second modeling task** (e.g. regression) — the rubric rewards depth on one task; the baseline → improved ladder across two targets is the intended breadth.
