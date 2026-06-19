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
