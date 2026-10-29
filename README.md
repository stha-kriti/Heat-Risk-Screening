# Computational Data Analysis

This TUM course project uses the FAIRUrbTemp urban air-temperature dataset to explore where summer heat exposure is highest across monitored city stations, and how useful geographic information is for screening local hotspots.

The analysis treats heat-risk screening as a descriptive and decision-support task. It is not a forecast, causal analysis, or recommendation for a specific intervention. Station coverage is uneven, and the available data does not include factors such as humidity, wind, or land cover; interpret results with those limits in mind.

## Project Deliverables

- [D1: Problem framing and exploratory data analysis](deliverable_1/d1.ipynb) audits the data and motivates the heat-exposure measures.
- [D2: Local-hotspot classification](deliverable_2/d2.ipynb) evaluates whether station geography can identify relatively hot locations within a city.
- [D3: Integrated analysis](deliverable_3/d3.ipynb) brings together absolute and within-city exposure, model validation, uncertainty, and screening implications.
- [Course notebooks](notebooks/) contain supporting exercises and demonstrations on Python, visualization, modeling, and related topics.

Design notes and implementation plans for the deliverables are kept alongside their notebooks.

## Data

The curated teaching subset is under [`archive/data/`](archive/data/). It contains hourly temperature data for 9 cities and metadata for 12 cities; metadata is also provided for some cities without hourly data in this subset. The deliverables read from this shared archive rather than copying the data into each folder. Generated parquet caches are stored under each deliverable's `cache/` directory.

The data originates from [FAIRUrbTemp](https://doi.org/10.6084/m9.figshare.29994877). It is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Please cite the original FAIRUrbTemp publication and acknowledge the curated teaching subset when reusing the data. See [the archive notes](archive/README.md) for attribution and coverage details.

## Setup

Use Python 3.10 or newer. From the repository root, create and activate a virtual environment, then install the project dependencies:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, create and activate the environment, then install dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Start JupyterLab from the repository root:

```sh
python -m jupyter lab
```

Open a deliverable notebook and run its cells in order. The analysis notebooks use the shared `archive/` directory and may create local cache files. The dependency list is maintained in [`requirements.txt`](requirements.txt).

## Repository Layout

```text
archive/          Curated FAIRUrbTemp data and dataset documentation
deliverable_1/    Problem framing and exploratory analysis
deliverable_2/    Local-hotspot classification
deliverable_3/    Integrated final analysis
notebooks/        Supporting course notebooks and demonstrations
```