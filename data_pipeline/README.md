# Module 1 — Data Pipeline (`/data_pipeline`)

Raw-to-relational catalogue pipeline: **scrape → clean → convert → store → query**.

## How to run

From inside this folder, in order:

```bash
pip install -r ../requirements.txt     # one consolidated requirements file at repo root
python scrape_books.py                 # needs internet (books.toscrape.com)
python build_db.py                     # builds books.db from scratch
python run_queries.py                  # runs the SQL suite + pandas equivalence check
```

| File | What it does |
|---|---|
| `scrape_books.py` | Scrapes 5 categories, cleans the fields, applies the fixed-rate conversion, writes `books_raw.csv` and `books_clean.csv` |
| `build_db.py` | Creates `books.db` from scratch (two-table PK/FK schema) and loads the cleaned CSV |
| `run_queries.py` | Executes 6 SQL queries, logs them with output to `outputs/query_results.md`, and proves `pd.merge` reproduces the SQL `JOIN` |
| `outputs/query_results.md` | Generated log: every query string with its printed output |

`books.db` is regenerated from scratch by `build_db.py`, so the repository ships
the exact recreation script rather than depending on a committed binary.

## Scope of the scrape

Categories scraped: **Fiction, Mystery, Historical Fiction, Romance, Nonfiction**
(5 categories ≥ the required 3), following each category's pagination to the
last page. Combined this yields well over the required 60 rows.

Captured per book: `title`, `price` (as listed, GBP), `star_rating` (as text,
e.g. "Three"), `availability` (as listed text), `category`.

## Cleaning decisions

| Field | Raw form | Cleaned form | Failure policy |
|---|---|---|---|
| `price` | `£51.77` | `price_gbp` — `float` | **Median imputation.** Price drives every downstream aggregation and the catalogue's price spread is tight, so a median stand-in keeps the row usable without distorting per-category averages. |
| `star_rating` | `"Three"` | `rating` — `int` 1–5 | **Median imputation**, same reasoning: the rating scale is a bounded 1–5 ordinal, so the median is a safe central stand-in. |
| `availability` | `"In stock (22 available)"` | `in_stock` — `bool` | **Drop the row.** There is no meaningful "median availability"; imputing it would invent stock state, which is exactly the kind of fabricated fact a competitive-intelligence feed must not contain. |
| `title` | text | text | **Drop the row** — a catalogue record with no title is not usable. |

The cleaning step prints a counts report (rows scraped, parse failures per
field, rows kept) so the effect of these rules is visible on every run, and the
pipeline never crashes on an unexpected row.

## Currency conversion

`price_inr = price_gbp × 105.50`, using the project's **fixed baseline rate of
1 GBP = 105.50 INR**. This is an artificial, project-defined constant for this
assignment — not a live or historical market rate — so it requires no API call,
no network access and no date reference. The rate lives in one place,
`GBP_TO_INR` in `scrape_books.py`.

The optional keyless-API stretch was **not** attempted; the graded fixed-rate
path is the only conversion path in the code.

## Schema

```sql
CREATE TABLE categories (
    category_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT NOT NULL UNIQUE
);

CREATE TABLE books (
    book_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT    NOT NULL,
    price_gbp   REAL    NOT NULL,
    price_inr   REAL    NOT NULL,
    rating      INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    in_stock    INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
    category_id INTEGER NOT NULL REFERENCES categories(category_id)
);
```

Normalized: the category name is stored once in the parent table and referenced
by foreign key from every book row, so renaming a category is a one-row update
and no book can reference a category that does not exist (`PRAGMA foreign_keys`
is enabled on load).

## SQL coverage

| Query | Clause(s) demonstrated |
|---|---|
| Q1 | `SELECT` + `WHERE` |
| Q2 | `ORDER BY` + `LIMIT` |
| Q3 | `DISTINCT` |
| Q4 | `IN` (plus a `JOIN`) |
| Q5 | `BETWEEN` |
| Q6 | `INNER JOIN` — the equivalence query |

## SQL vs pandas equivalence

Q6 is read back with `pd.read_sql(...)` and independently reproduced with
`pd.merge(books_df, categories_df, on="category_id", how="inner")` followed by
the same column selection and sort. `run_queries.py` prints both results side
by side, asserts `DataFrame.equals` between them, and records both in
`outputs/query_results.md`. A stable (`mergesort`) sort is used on the pandas
side so ties break in the same order SQLite produces.
