# Module 2 — Analytics Pipeline (`/analytics`)

One cohesive pipeline: the dataset is loaded once, cleaned once, and every
later step builds on that same data.

## How to run

```bash
pip install -r ../requirements.txt
python 01_eda.py        # Part A: the single raw load, profiling, cleaning, the data story
python 02_modeling.py   # Part B: split, pipeline, models, tuning, regression, persistence
```

## Files

| File | Role |
|---|---|
| `01_eda.py` | Part A, Tasks 1–6. Performs the module's **only** raw load. |
| `02_modeling.py` | Part B, Tasks 7–15. Reads `titanic.csv`; never calls `sns.load_dataset`. |
| `titanic.csv` | The committed offline fallback, written by `01_eda.py` via `df.to_csv("titanic.csv", index=False)` |
| `titanic_clean.csv` | The cleaned frame produced by Task 2 |
| `reports/01_eda_report.md` | Every measured number and written interpretation for Part A |
| `reports/02_modeling_report.md` | All metrics, comparison tables and conclusions for Part B |
| `figures/` | All charts (`.png`), referenced from the reports |
| `best_pipeline.joblib` | The fitted preprocessing + estimator, saved as one object |

## The single-load rule

`01_eda.py::load_once()` reads `titanic.csv` if it exists; otherwise it calls
`sns.load_dataset('titanic')` exactly once (network needed the first time,
seaborn caches thereafter) and immediately writes `titanic.csv`.
`02_modeling.py` reads that file. There is no second `sns.load_dataset` call
anywhere in the module, so the whole thing is gradable offline via
`pd.read_csv("titanic.csv")`.

## Where the written interpretations live

Both scripts assemble a Markdown report as they run and write it to
`reports/`. The numbers are measured at run time and the interpretive text is
authored, so the prose and the figures it cites can never drift apart. Read
`reports/01_eda_report.md` and `reports/02_modeling_report.md` for:

- missing-value percentages per column with the threshold rule that chose each strategy
- IQR outlier counts for `age` and `fare`, and the mean/median/mode skew conclusion
- survival rates by sex, by pclass, and by sex × pclass
- the 6×6 correlation matrix, all off-diagonal pairs ranked by `|r|`, and the top two interpreted
- five multivariate charts, each with its own interpretation
- the before/after standardisation check
- confusion matrices, the full metric suite and ROC/AUC for all three classifiers
- the three-way imbalance comparison and its conclusion
- GridSearchCV best parameters and the OOB score
- the regression metrics and the heteroscedasticity conclusion
- the final two-group comparison table and the deployment recommendation

> Run both scripts once before submitting: `figures/`, `reports/`,
> `titanic.csv` and `best_pipeline.joblib` are all generated artifacts.
