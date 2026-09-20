# Zepto Data & AI Platform — Capstone

One repository, three connected modules: a data-engineering pipeline that turns
raw scraped data into a clean relational store, an analytics pipeline that
profiles and models a customer-style dataset end to end, and a GenAI support
assistant that answers policy questions grounded in Zepto's own documents.

| Module | Path | Marks | What it ships |
|---|---|---|---|
| 1 — Data Pipeline | [`/data_pipeline`](data_pipeline/) | 25 | scrape → clean → GBP→INR → normalized SQLite → SQL + pandas |
| 2 — Analytics Pipeline | [`/analytics`](analytics/) | 50 | profiling, cleaning, EDA story, 3 classifiers, tuning, regression, saved pipeline |
| 3 — Support Assistant | [`/support_assistant`](support_assistant/) | 25 | ChromaDB RAG + LangGraph router + Pydantic schema + FastAPI + Docker |

Each module also has its own README with the detail; this page covers setup,
how to run everything, and the design decisions.

---

## Setup

**Dependencies: one consolidated `requirements.txt` at the repository root**
covers all three modules for local development. The only exception is
`support_assistant/requirements.txt`, a slim subset used by the Docker image so
the container does not install pandas, scikit-learn and seaborn it never uses.

```bash
git clone <this-repo-url>
cd zepto-ai-capstone

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+ (3.11 recommended).

### What needs the network

| Step | Needs network? |
|---|---|
| `data_pipeline/scrape_books.py` | **Yes** — scrapes `books.toscrape.com` (public, no key, no paid tier) |
| `analytics/01_eda.py` | Only on the very first run, and only if `analytics/titanic.csv` is absent. Once that file exists the module reads it from disk. |
| `analytics/02_modeling.py` | No — reads the committed `titanic.csv` |
| `support_assistant` | Only to download the `all-MiniLM-L6-v2` weights once. **No LLM API call is made at all on the graded path.** |

---

## Running each module

### Module 1 — Data Pipeline

```bash
cd data_pipeline
python scrape_books.py     # scrape 5 categories, clean, convert -> books_clean.csv
python build_db.py         # build books.db from scratch (two-table PK/FK schema)
python run_queries.py      # 6 SQL queries + pd.read_sql/pd.merge equivalence check
```

Query strings and their output land in `data_pipeline/outputs/query_results.md`.

### Module 2 — Analytics Pipeline

```bash
cd analytics
python 01_eda.py           # single raw load -> titanic.csv, profiling, cleaning, EDA story
python 02_modeling.py      # reads the same titanic.csv, trains/tunes/evaluates, saves the pipeline
```

Written interpretations, every measured number and the comparison tables are
written to `analytics/reports/01_eda_report.md` and
`analytics/reports/02_modeling_report.md`; charts go to `analytics/figures/`;
the fitted end-to-end pipeline is saved as `analytics/best_pipeline.joblib`.

### Module 3 — Support Assistant

```bash
cd support_assistant
python ingest.py                       # embed the 8 docs into ChromaDB
python graph.py                        # demo: one query routed each way
uvicorn main:app --reload --port 7860  # serve POST /ask
python demo_calls.py                   # record the example calls into the README
```

Or containerized:

```bash
docker build -t zepto-support .
docker run -p 7860:7860 zepto-support
```

**Leave `MOCK_LLM` unset.** Unset (or `1`) is the deterministic offline mock
path, and that is what is graded.

---

## Design decisions

### Module 1 — Data Pipeline

- **Scope.** Five categories (Fiction, Mystery, Historical Fiction, Romance,
  Nonfiction), following each category's pagination to its last page — well
  past the required 60 rows across ≥ 3 categories.
- **Cleaning policy is split by column type, not applied uniformly.** Numeric
  fields (`price_gbp`, `rating`) use **median imputation** on a parse failure,
  because price drives every downstream aggregation and a bounded 1–5 rating
  has a safe central stand-in. `in_stock` and `title` failures **drop the
  row**: there is no meaningful median availability, and imputing it would
  invent stock state — precisely the fabricated fact a competitive-intelligence
  feed must not carry. The pipeline prints a counts report every run so the
  effect is visible rather than silent.
- **Currency.** `price_inr = price_gbp × 105.50`, using the project's **fixed
  baseline rate of 1 GBP = 105.50 INR**. This is an artificial, project-defined
  constant for this assignment — not a live or historical market rate — so it
  requires no API call, no network access and no date reference. The optional
  keyless-API stretch was not attempted.
- **Schema.** Normalized into `categories` (parent, `category_name UNIQUE`) and
  `books` (child, `category_id` FK), with `CHECK` constraints on `rating` and
  `in_stock` and `PRAGMA foreign_keys=ON` during load. The repository ships the
  exact regeneration script rather than relying on a committed binary.
- **Equivalence check.** The join query is read back with `pd.read_sql` and
  independently reproduced with `pd.merge`; the script asserts
  `DataFrame.equals` between them and prints both side by side. A stable
  (`mergesort`) sort is used on the pandas side so ties break in SQLite's order.

### Module 2 — Analytics Pipeline

- **One load, one lineage.** `01_eda.py` performs the module's only raw load:
  if `titanic.csv` exists it is read from disk, otherwise
  `sns.load_dataset('titanic')` is called once and immediately written to
  `titanic.csv`. `02_modeling.py` reads that same committed file and never
  calls the seaborn loader, so the whole module touches the network/cache
  exactly once and is fully gradable offline.
- **Missing values by measured threshold.** Every affected column's exact
  missing percentage is measured at run time and printed next to the strategy
  the threshold rule dictates (< 5% → drop rows, 5–30% → impute, > 30% →
  explicit decision). `deck`, at roughly three-quarters missing, is **dropped
  rather than encoded as its own category**: its surviving values are not
  missing at random — they skew to upper-class passengers — so a "missing"
  level would mostly re-encode `pclass`, which the feature set already has.
- **Correlation matrix on exactly six columns.** `survived`, `pclass`, `age`,
  `sibsp`, `parch`, `fare`. `adult_male` and `alone` are excluded as derived
  flags computable from `sex`/`age` and `sibsp`+`parch`. All off-diagonal pairs
  are ranked by `abs(r)` and the top two are named and interpreted in text.
- **Leakage handled in two places.** (a) The raw frame's `alive` column is a
  verbatim copy of the target and is excluded from the feature set along with
  the other derived restatements (`class`, `who`, `adult_male`, `embark_town`,
  `alone`). (b) Every preprocessing step lives inside a `ColumnTransformer`
  wrapped in a `Pipeline`, so `fit()` is called exactly once, on the training
  split — the test split can only ever be transformed. The fit/transform
  separation is structural, not a rule anyone has to remember.
- **The EDA standardisation check is deliberately separate.** Task 6's z-score
  comparison runs on the full cleaned frame purely as a sanity check; it does
  not feed the model, which fits its own scaler on training data only.
- **SMOTE is confined by construction.** The oversampling variant uses
  `imblearn.pipeline.Pipeline`, which applies a sampler during `fit` only and
  never during `predict`/`transform`, so the test fold is never synthesised on.
- **OOB is available because it was requested up front.** The tuned forest is
  constructed as `RandomForestClassifier(oob_score=True, bootstrap=True, …)`;
  `oob_score_` is not populated otherwise.
- **The saved artifact is the whole pipeline.** `joblib.dump` persists the
  fitted `ColumnTransformer` *and* the tuned estimator as one object, so it
  consumes raw, unpreprocessed rows end to end. The reload check feeds it raw
  input — including a row with a missing `age` and `embarked` — and confirms it
  reproduces the in-memory model's test predictions exactly.
- **Metric groups stay separate.** The final table presents classification
  metrics (accuracy/precision/recall/F1/AUC) and regression metrics
  (MAE/RMSE/R²/Adjusted R²) as two distinct groups, never merged onto one scale.

### Module 3 — Support Assistant

- **The offline mock is the product, not a stub.** `MOCK_LLM` unset (or `1`) is
  the default and the graded path: deterministic keyword routing, a canned
  templated answer built from the top retrieved chunk, and a schema populated
  from code. No signup, no API key, no LLM network call. `MOCK_LLM=0` is an
  implemented but unexercised optional extension.
- **Retrieval is never mocked.** Embedding with `all-MiniLM-L6-v2` and top-3
  cosine search in ChromaDB run for real in both modes, so the graded path
  still demonstrates that a returns question actually retrieves the returns
  document.
- **Routing is independent of the toggle.** The conditional edge reads
  `state["intent"]`; only the generation step inside each node branches on
  `MOCK_LLM`. Swapping the LLM backend cannot change the graph topology.
- **Per-document chunking.** Each policy is one short paragraph; splitting it
  would separate a rule from its exception (e.g. `doc_06`'s INR 1000 photo
  requirement), which is exactly the half-fact a grounded assistant must not
  serve.
- **Docker as the graded baseline.** The image bakes in the embedding weights
  and runs `ingest.py` at build time, then serves `POST /ask` on `0.0.0.0:7860`
  with `MOCK_LLM` left unset. No Hugging Face Spaces deployment was attempted.

---

## Git workflow

Scored once against the whole repository. The history contains a feature branch
(`feature/support-assistant`) created from `main`, committed to more than twice,
and merged back into `main` with an explicit merge commit (`--no-ff`):

```bash
git log --graph --all --oneline
```

---

## Repository layout

```
.
├── README.md                     <- you are here
├── requirements.txt              <- consolidated, all three modules
├── data_pipeline/
│   ├── README.md
│   ├── scrape_books.py           scrape + clean + fixed-rate conversion
│   ├── build_db.py               two-table PK/FK SQLite schema + load
│   ├── run_queries.py            6 SQL queries + pd.merge equivalence
│   └── outputs/query_results.md  generated query log
├── analytics/
│   ├── README.md
│   ├── 01_eda.py                 the single raw load, profiling, cleaning, EDA
│   ├── 02_modeling.py            split, pipeline, 3 models, tuning, regression
│   ├── titanic.csv               committed offline fallback (generated once)
│   ├── figures/                  all charts
│   └── reports/                  generated markdown with every interpretation
└── support_assistant/
    ├── README.md
    ├── docs/doc_01.txt … doc_08.txt
    ├── ingest.py                 chunk + embed + ChromaDB + retrieve()
    ├── prompts.py                role/context/task/format/length templates
    ├── schemas.py                Pydantic request + answer schema
    ├── llm.py                    MOCK_LLM toggle + retry-on-invalid
    ├── graph.py                  LangGraph: 3 nodes + conditional edge
    ├── main.py                   FastAPI POST /ask
    ├── demo_calls.py             records the example calls
    ├── requirements.txt          slim set for the container
    └── Dockerfile
```

## Academic integrity

The code, analysis and written interpretations in this repository are my own.
Library and framework documentation was consulted; the reasoning and
implementation are authored by me.
