# D1 Planning — Discussion Document

**For:** Kriti + partner (group of 2)
**Deliverable:** D1, due **May 10, 2026** (~2 days). Weight: 15%.
**Working notebook:** `deliverable_1/d1.ipynb`.
**Dataset:** FAIRUrbTemp curated subset, co-located at `deliverable_1/archive/` (Amini et al., *Scientific Data*, 2026).

This document is the input for our D1 alignment conversation. It is intentionally **not** an implementation. By the end of the discussion we should have decided: (1) the track, (2) the scope, (3) the task split, and (4) what each of us writes by when.

---

## 1. What the course is actually asking of D1

The instructor framing across Session 1 (Data Analysis 101), Session 3 (Data Sources & Quality), and Session 4 (EDA) makes the expected shape of D1 very specific. It is **not** "explore the data and make plots." It is:

> Climb the DIKD pyramid (Data → Information → Knowledge → Decision) for **the first two rungs**, with the explicit motivation of a decision the analysis will support.

This is reinforced by the D1 rubric, which gives equal 30% weights to:
- **Problem formulation** — clear, specific, measurable, *actionable*. ("Vague questions produce misleading analyses." — S01)
- **Data understanding & quality checks** — six dimensions of quality (S03): accuracy, completeness, consistency, timeliness, validity, uniqueness. Plus measurement scale (nominal/ordinal/interval/ratio) and systematic vs. random error.
- **Exploratory data analysis** — every figure must serve at least one of the **four purposes** of EDA (S04): **Describe, Diagnose, Discover, Decide**. Connected to the question, not "plotting for the sake of plotting" (`project_guidelines.md`).
- (And 10% communication/structure.)

**What this means for us, in plain terms:** D1 is graded on whether we can frame a real decision question on this dataset, audit the data honestly, and produce *small, purposeful* visualizations that lead toward a model in D2. Modelling is explicitly out of scope for D1.

**Most relevant tutorial notebooks to mirror in style:** `T3_from_missing_to_insights*` (cleaning + missing-data audit) and `T4_from_data_points_to_visualization*` (EDA figures). They are the implicit "what good looks like."

---

## 2. The dataset, in numbers

I scanned one station file per city (full 12-line SEF header + tabular `Year/Month/Day/Hour/Minute/Period/Value/Meta`). Real numbers from `deliverable_1/archive/data/hourly/`:

| City        | # Stations | Sample rows (1 file) | NA rate (1 file) | Year range (1 file) |
|-------------|-----------:|---------------------:|-----------------:|---------------------|
| Amsterdam   |        23  |              84,046  |            48.8% | 2014–2023           |
| Basel       |       215  |              26,305  |            39.2% | 2019–2022           |
| Berlin      |        11  |              35,088  |            45.0% | 2020–2024           |
| Bern        |        55  |              11,880  |             0.2% | 2019–2022           |
| Biel        |        39  |               2,975  |             0.3% | 2023                |
| Birmingham  |        23  |              24,805  |            54.0% | 2019–2022           |
| Freiburg    |        44  |              15,267  |            21.4% | 2022–2023           |
| Novi Sad    |        26  |              35,991  |            11.8% | 2013–2018           |
| Zurich      |       276  |              21,911  |            40.3% | 2019–2021           |
| **Total**   |    **712** |                      |                  |                     |

(Three more cities — Ghent, Rennes, Turku — have metadata only, no hourly data.)

### Three observations that should drive every D1 decision

1. **Coverage is wildly unequal across cities.** Zurich has 276 stations, Berlin only 11. Amsterdam covers 10 years; Biel only 1; Novi Sad covers 2013–2018 (no overlap with most others). **Implication:** any cross-city comparison must hold time period and station selection constant, or report results stratified by both. Naive comparisons will be misleading (S03 sampling-bias trap).
2. **Missing-data rates are extreme and very different across cities.** Bern is essentially complete (~0.2% NA in the sample). Birmingham is half-missing (54%). Amsterdam, Basel, Berlin, Zurich are all 39–49% NA. **Implication:** the *missing-data audit itself* is one of the strongest D1 figures we can produce — it directly feeds the "data understanding" rubric (30%) and the S04 "Diagnose" purpose. We don't need to fix it, just describe it honestly.
3. **Only one variable is recorded: `ta` (air temperature, °C).** Plus station metadata (lat, lon, sensor height) and city-level network info. **Implication:** the project is really about how we *aggregate* and *threshold* `ta` to construct a derived target, not about predicting `ta` from many features. This narrows the realistic track choices.

### The natural common time window
Looking across cities, **2019–2022 (4 years, summers)** is the window that intersects with the most cities (Amsterdam, Basel, Bern, Birmingham, Freiburg, Novi Sad partial, Zurich). Berlin starts 2020. This is the de-facto study period the data is curated around — Bern is explicitly described in the dataset README as "summer 2019 to 2022." We should default to this window unless we have a reason not to.

---

## 3. Track tradeoffs (deep on T1 and T2, brief on T3/T4)

We're leaning T1/T2. Here's what each actually looks like in D1, D2, D3, mapped to what the data supports.

### Track 1 — Heat-risk screening *("Where is heat risk highest?")*

**D1 shape:**
- Define a heat-risk metric (candidate: hours per summer with `ta` ≥ X°C; or 95th percentile of summer daily max).
- EDA: cross-city / cross-station temperature distributions, summer time-series, missing-data audit, threshold-exceedance maps using lat/lon from metadata.
- Frame the *decision*: where should mitigation be prioritized? (S01 Stage 4: interpretation.)

**D2 shape:** Compute the metric properly. **Sensitivity analysis** is the "method": vary the threshold (e.g., 28°C / 30°C / 32°C), vary the time window, vary the station-completeness filter, see how the ranking of stations/cities changes. Bootstrap confidence intervals on the metric.

**D3 shape:** Final ranking + recommendation, with explicit uncertainty.

**Pros for novice + 2-day D1:**
- Lowest math/ML burden of any track. Everything is `pandas` aggregation + plots — exactly the T3/T4 tutorial style.
- Heat-risk metric is intuitive to motivate to instructors and to the partner.
- The big EDA figures (cross-city distributions, missing-data heatmaps, station-rank tables) are *the deliverable*, not preparation for one.

**Risks:**
- "Where's the model?" trap in D2. We must deliberately frame the metric+ranking as the method and the sensitivity/bootstrap analysis as the validation, otherwise D2 looks thin.
- Lower ceiling for striking results — clean story, not a flashy one.

---

### Track 2 — Heat alert system *("Can we predict high-heat events?")*

**D1 shape:**
- Define a binary target (candidate: "next-hour `ta` ≥ 30°C" or "tomorrow's daily max ≥ threshold").
- EDA: target base rate over time (per city, per season, per hour-of-day), class balance, distribution of `ta` lags as candidate predictors.
- Frame the *decision*: what alert threshold balances missed events vs. false alarms? (Sharp decision question — strong fit for D3 25%-weighted "decision insight" rubric.)

**D2 shape:** Train a classification model (logistic regression baseline → maybe a tree). **Time-aware** train/test split — never random. Tune decision threshold. Report precision, recall, ROC, confusion matrix. Class imbalance handling.

**D3 shape:** Final precision-recall trade-off curve + recommended threshold + cost framing.

**Pros:**
- D2 is **the most rubric-aligned** of any track — D2 weights validation 30% and explicitly mentions "metrics" and "error analysis"; classification gives confusion matrices, ROC, threshold curves on a silver platter.
- D3's "decision insight" 25% maps perfectly to "which alert threshold to use".
- Logistic regression is a beginner-friendly, transparent baseline; sklearn does the heavy lifting.

**Risks:**
- More moving parts: target definition, **time-based splitting** (random splits leak future info — common pitfall mentioned in S04's "garden of forking paths"), threshold tuning, imbalanced-class metric selection.
- We'd both need to commit to the temporal-split discipline early. Not hard, but easy to forget.

---

### T3 (Drivers) and T4 (Monitoring) — short version

- **T3 — Drivers:** like T2 but interpretation-heavy (feature importance, multicollinearity, partial dependence). With only `ta` + thin metadata, "drivers" are mostly time-of-day, season, station altitude, sensor height, lat. Doable but lighter feature space than T3 typically assumes. **Skip unless we both find the "why" framing more motivating than "what".**
- **T4 — Monitoring & representativeness:** methodologically clever (rolling windows, bootstrap stability), but D1's standard "describe + diagnose" EDA template is awkward to fit here — the EDA *is* the modelling, which is unusual and harder to present cleanly under time pressure. **Skip given 2-day D1 deadline.**

---

### My honest read for our situation

**T1 is the safer floor; T2 is the higher ceiling.**

- T1 is genuinely hard to fail at if we're careful. The risk is a thin D2.
- T2 maps better to the D2 and D3 rubrics, but we need temporal-split discipline and we accept a real modelling step in D2. Total work is ~30% more, but the deliverable quality ceiling is meaningfully higher because the alert-threshold trade-off makes a sharp D3 decision story.
- They are **not mutually exclusive at the D1 stage.** A heat-risk metric (T1's core artifact) is also useful as the basis for T2's binary target. So D1 can be ~80% shared between them; the track choice fully forks at D2.

**My recommendation:** *start with the intent of T2, but design D1 so it would also satisfy T1.* This keeps optionality. We can confirm or switch by end of D1.

If we'd rather lock in T1 now to reduce stress, that's also defensible — it's the lower-variance choice and the EDA chapters of D1 (cross-city distributions, missing-data audit, station rankings) are already strong enough on their own for a good D1 grade.

---

## 4. D1 EDA plan — mapped to the four purposes (S04) and the rubric

Session 4 is explicit: every plot serves **Describe, Diagnose, Discover, or Decide**. We should plan our notebook section-by-section against this. Below: the figures I think we should aim for. Each is annotated with its **purpose**, the **rubric category** it earns points in, and which **track** it serves.

| #  | Figure / table                                                     | Purpose       | Rubric                         | Track     |
|---:|--------------------------------------------------------------------|---------------|--------------------------------|-----------|
| 1  | **Coverage table:** city × n_stations × time-range × % NA          | Diagnose      | Data understanding (30%)       | T1, T2    |
| 2  | **Missing-data heatmap** (city × month, % NA)                      | Diagnose      | Data understanding (30%)       | T1, T2    |
| 3  | **Per-city `ta` distribution** (boxplot or ridgeline, summer only) | Describe      | EDA (30%)                      | T1, T2    |
| 4  | **Diurnal cycle:** mean `ta` by hour-of-day, faceted by city       | Describe      | EDA (30%)                      | T1, T2    |
| 5  | **Annual cycle:** mean `ta` by month, faceted by city              | Describe      | EDA (30%)                      | T1, T2    |
| 6  | **Within-city station spread** (box of stations within Basel/Zurich) | Discover    | EDA (30%)                      | T1 strong |
| 7a | **(T1)** Threshold-exceedance ranking: stations × hours ≥ 30°C     | Decide        | Problem formulation (30%) + EDA (30%) | T1 |
| 7b | **(T2)** Target base rate over time (binary high-heat events)      | Decide        | Problem formulation (30%) + EDA (30%) | T2 |
| 8  | **Station map** (lat, lon, sensor height) for 1–2 focus cities     | Describe      | EDA (30%)                      | T1, T2    |
| 9  | **Markdown:** problem statement, decision frame, success criteria  | (text)        | Problem formulation (30%) + Communication (10%) | both |
| 10 | **Markdown:** known limitations & biases (S03 sampling/quality)    | (text)        | Data understanding (30%) + Communication (10%) | both |

**That's roughly 7 figures + 2 well-written markdown sections.** S04 explicitly cites a "three-plot EDA report" exercise where each plot must serve a different one of the four purposes — we'll exceed that, but the principle holds: **no plot without a purpose tied to the question.**

**Stylistic constraints (S04 cheatsheets):**
- Use `viridis` / Set2 / RdBu_r — never `jet`/rainbow.
- Position > color > area in encoding choice. So: bar charts and dot plots for rankings (not pie); small multiples for per-city panels.
- Lie factor ≈ 1 (don't truncate y-axes). Data-ink ratio: erase chartjunk.
- Each figure has **readable labels and units** (`°C`, hour-of-day, station IDs).

### Problem-statement template (for the markdown cells)

For both T1 and T2, the markdown sections should explicitly state:
- **Question** (one sentence, decision-oriented). e.g., *"Among the {N} stations across {cities} during summer 2019–2022, which experience the highest heat exposure, and how stable is that ranking?"*
- **Target / outcome** (precisely defined: which variable, what aggregation, what threshold).
- **Inputs / features available** (`ta` time series, station metadata: lat, lon, sensor height, city, network operator).
- **Intended use** (S01 framing: prediction? screening? prioritization?).
- **Success criteria** (what would make the analysis useful?).
- **Assumptions and known limitations** — sensors are 2–4 m street-level, urban settings only, 2019–2022 window, NA rates differ by city, no causal claims (S01: "correlation tells you what; causation tells you why").

---

## 5. Task split for 2 people

The work has natural seams. A reasonable split for D1 (each person does ~equal hours):

### Person A — "Data engineer" (Kriti?)
1. Write a small `load_sef.py` helper module: parse the 12-line SEF header, return a tidy `DataFrame` with columns `city, station_id, datetime, ta, meta_flag`. Cache once to a single `.parquet` so we don't re-parse 700+ files every notebook run.
2. Compute the **coverage table** (figure #1) and **missing-data heatmap** (figure #2) — these are the data-quality backbone.
3. Write the data-quality + limitations markdown (figure #10) using S03 vocabulary (completeness, validity, sampling frame, systematic vs. random error).
4. Reproducibility check: kernel-restart, re-run top-to-bottom, confirm no manual steps.

### Person B — "Analyst"
1. Write the **problem statement** markdown (figure #9). Define the heat-risk metric or binary target precisely.
2. Build the four core EDA figures: distribution by city (#3), diurnal cycle (#4), annual cycle (#5), within-city spread (#6).
3. Build the **track-specific decision figure** (#7a or #7b depending on what we land on).
4. Write each figure's caption + 2–3 sentence interpretation. *Every figure has a sentence saying what it shows and why we care.*

### Joint
- Track decision call: **before any code** — schedule a 30-min sync to lock T1 vs. T2 vs. T1+T2-keep-options-open.
- Notebook integration: one of us merges everything into `d1.ipynb` with consistent section headings mapped to the rubric (Problem framing → Data understanding → EDA → Limitations).
- Final read-through together before submission.

---

## 6. Two-day timeline to May 10

Today is **2026-05-09**. Submission **May 10**.

| When                     | Who      | Output                                           |
|--------------------------|----------|--------------------------------------------------|
| **May 9 morning** (now)  | Both     | Read this doc, agree track, agree split          |
| May 9 afternoon          | A        | `load_sef.py` + cached parquet of all hourly data |
| May 9 afternoon          | B        | Draft problem statement markdown + heat-risk metric definition |
| May 9 evening            | A        | Coverage table + missing-data heatmap finalized  |
| May 9 evening            | B        | Distribution + diurnal + annual cycle figures    |
| **May 10 morning**       | A        | Limitations markdown + reproducibility re-run    |
| May 10 morning           | B        | Track-specific decision figure (#7) + within-city spread |
| May 10 midday            | Joint    | Merge into `d1.ipynb`, re-run from clean kernel   |
| May 10 afternoon         | Joint    | Final read-through, submit                        |

Slack: keep figures to ~7 (don't over-build). Communication is 10% of the grade — **a clean, focused notebook beats an ambitious sprawling one.**

---

## 7. Open questions to resolve in the partner discussion

These are decisions only you both can make. I need answers before I can scaffold the notebook.

1. **Track:** T1 (safer), T2 (higher ceiling), or T1+T2-shared D1 (recommended). Pick one.
2. **Scope:**
   - All 9 cities in the coverage table & cross-city distributions (yes — needed for D1 strength), AND
   - 3–4 representative cities for deeper EDA (recommended), OR all 9 (more work). Suggested 4: **Amsterdam** (coastal-mild, long history), **Berlin** (continental), **Novi Sad** (warm-Pannonian), **Birmingham** (UK-maritime). Climate diversity is the justification.
3. **Time window:** 2019–2022 summer (June–August)? This is the natural common window. Alternative: full available period per city (more honest about coverage but harder to compare).
4. **Heat-risk metric (or target):** if T1, hours-above-threshold or percentile-based? If T2, next-hour binary or daily-max binary? At what threshold (28 / 30 / 32 °C)?
5. **Tooling:** are we both comfortable with `pandas` + `matplotlib` + `seaborn`? Anyone want to use `plotly` for interactivity? (Recommendation: stick to matplotlib/seaborn for grading consistency with the tutorial style.)
6. **Repository discipline:** commit the cached `.parquet` or .gitignore it and re-derive each clone? (Recommendation: `.gitignore` it; the project guidelines say "do not commit large derived data files".)

---

## 8. What we will NOT do in D1 (and why)

- **No modelling.** D1 is framing + data + EDA. Modelling is D2. The project guidelines list "jumping into modeling before understanding the data" as a top common pitfall.
- **No causal claims.** With only `ta` and a few metadata columns, every relationship we observe is correlational. S01 is explicit: "correlation tells you WHAT; causation tells you WHY. Do not confuse them."
- **No silent fixes to the data.** If a station has 54% missing values, we *report* it, we don't impute it away. (Imputation belongs to D2's preprocessing pipeline, with justification.)
- **No plotting just to fill space.** S04: "each figure should support an insight." If we can't write a one-sentence interpretation, the figure shouldn't be in the notebook.

---

*This document is a living artifact. Update it after the partner discussion with the decided track, scope, and any agreed deviations from this plan.*
