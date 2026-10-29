# D1 — Problem statement & motivation (v1, rough draft)

> **Track 1 — Heat-risk screening.** This is a first draft. Decisions marked **[CHOICE]** are placeholders we should debate together; ones marked **[VERIFY]** depend on what the EDA reveals.

---

## Motivation

European cities are warming faster than the global average, and street-level heat exposure is increasingly recognised as a public health risk — particularly during summer. Heat is not uniform within a city: a sensor on a tree-lined residential street and a sensor on an asphalt square a kilometre away can record very different temperatures during the same heatwave. For a city planner deciding where to install shade structures, plant trees, or open cooling centres, a city-average temperature is too coarse to act on. What they need is a **station-level (or neighbourhood-level) ranking of heat risk**, plus an honest sense of how much that ranking depends on the choices made in the analysis.

The FAIRUrbTemp dataset (Amini et al., 2026, *Scientific Data*) provides exactly this kind of input: street-level (2–4 m) air-temperature measurements across multiple European cities, quality-controlled through a 7-step pipeline. The curated subset we use covers **9 cities and ~712 stations**, with hourly readings concentrated in the 2019–2022 summers. This is the empirical basis on which a heat-risk screening can be built.

## Question

> **Among the stations in the FAIRUrbTemp curated subset (2019–2022 summers), which locations experience the highest heat exposure, and how stable is that ranking under different reasonable choices of metric, threshold, and time window?**

This question is intentionally narrow. We are not asking *why* some stations are hotter than others, nor whether they will be hotter in the future. We are asking, descriptively, **where the data we have points to the highest exposure**, and how confident we should be in that pointing.

## Target / outcome of interest

The outcome is a **per-station heat-risk score** computed from the hourly air-temperature time series `ta` (°C). Two candidate definitions, both standard in the heat-exposure literature:

- **[CHOICE A] Threshold exceedance:** number of hours per summer with `ta` ≥ T °C, where T is a fixed threshold (e.g., 30 °C). Easy to interpret; sensitive to the choice of T.
- **[CHOICE B] Distributional:** the 95th (or 99th) percentile of summer hourly `ta` per station. Less sensitive to a single threshold; harder to communicate.

D1 will report both and compare. D2 will pick one as the headline metric, with the other used for sensitivity analysis.

## Inputs / features

- **Hourly `ta` (°C)** per station, from the SEF-format TSVs in `archive/data/hourly/<City>/`.
- **Station metadata:** `station_id`, latitude, longitude, sensor height (typically 2–4 m).
- **Network metadata:** city, network operator, geographic coverage.
- **Measurement metadata:** sensor type, measurement interval, QC flag.

We do not have weather covariates (wind, humidity, solar radiation) or land-cover features in this subset. Heat-risk is therefore defined purely from the temperature record, not as a multi-factor index.

## Intended use (decision frame)

The analysis is a **screening tool**. The output is a ranked list of stations (and aggregated to neighbourhoods or cities where appropriate) with associated uncertainty, intended to support a decision of the form:

> *"Given a fixed budget for heat-mitigation interventions across these European cities, which 10–20 locations should be considered first, and how robust is that selection?"*

It is **not** a forecast, a causal attribution, or a recommendation for any specific intervention. It tells a planner *where to look further*, not *what to do*.

## Success criteria for D1

D1 will be considered successful if it:

1. Defines a heat-risk metric precisely enough that a third party could re-compute it from `archive/`.
2. Audits the data honestly — coverage by city, missing-data rates, time-period overlap, sensor heterogeneity.
3. Produces a first descriptive picture (distributions, diurnal/annual cycles, within-city spread) that motivates the metric and surfaces likely pitfalls before D2.
4. Names its limitations explicitly.

A quantitative ranking of stations is **not** required in D1 — that is D2's preprocessing + sensitivity work. D1 only needs to make the case that such a ranking is well-defined and that the data supports computing it.

## Assumptions and known limitations

- **Sampling frame is opportunistic, not representative.** The 9 cities are those for which COST Action FAIRNESS partners contributed quality-controlled data; they are biased toward central / Western European urban climates. Conclusions cannot be generalised to "European cities" as a whole.
- **Within-city station placement is not random.** Sensors were installed by research networks for their own purposes (e.g., urban heat-island studies). Some neighbourhoods are densely instrumented; others have no sensors. We should not interpret the absence of high-risk stations in an area as the absence of heat risk.
- **Time coverage is uneven.** Some cities have data spanning a decade; others only one summer. Cross-city comparisons must hold the time window constant. **[VERIFY]** common window after EDA — current best guess is summer 2019–2022.
- **Missing-data rates are extreme and variable across cities** (preliminary check: from ~0.2% in Bern to ~54% in Birmingham in sample files). The ranking must explicitly account for completeness, e.g., by requiring a minimum data availability threshold per station.
- **Only one variable is measured.** No humidity, no wind, no land cover. "Heat risk" here is defined narrowly as exposure to high air temperature, not the full thermal-comfort or health-risk picture.
- **No causal claims.** Differences between stations may reflect microclimate, sensor placement, sensor model, or measurement artefacts. We can describe; we cannot attribute.

---

## Open questions for partner review (drop these once decided)

1. **Choice of metric** (A: hours-above-threshold vs. B: percentile vs. both).
2. **Threshold value** if we go with A — 28, 30, or 32 °C? Or report a curve over thresholds?
3. **Time window** — strict 2019–2022 summers (June–August)? Or all available summers per station, with caveats?
4. **Granularity of the final ranking** — stations? Neighbourhoods? Cities? All three?
5. **Minimum data-availability filter** — drop stations with > X% NA in the study window? What X?
