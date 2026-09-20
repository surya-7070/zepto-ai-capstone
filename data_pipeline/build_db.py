"""
Module 1, step 2: build the normalized SQLite store.

Creates books.db from scratch with a two-table PK/FK schema and loads
books_clean.csv into it.

    categories(category_id PK, category_name UNIQUE)
    books(book_id PK, title, price_gbp, price_inr, rating, in_stock,
          category_id FK -> categories.category_id)

Run:  python build_db.py
"""

from __future__ import annotations

import os
import sqlite3

import pandas as pd

DB_PATH = "books.db"
CSV_PATH = "books_clean.csv"

SCHEMA = """
DROP TABLE IF EXISTS books;
DROP TABLE IF EXISTS categories;

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

CREATE INDEX idx_books_category ON books(category_id);
"""


def build(db_path: str = DB_PATH, csv_path: str = CSV_PATH) -> None:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found - run `python scrape_books.py` first."
        )

    df = pd.read_csv(csv_path)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        conn.executescript(SCHEMA)

        # --- categories (parent table) -----------------------------------
        categories = sorted(df["category"].unique())
        conn.executemany(
            "INSERT INTO categories (category_name) VALUES (?);",
            [(c,) for c in categories],
        )

        lookup = dict(
            conn.execute("SELECT category_name, category_id FROM categories;").fetchall()
        )

        # --- books (child table, FK to categories) ------------------------
        rows = [
            (
                r.title,
                float(r.price_gbp),
                float(r.price_inr),
                int(r.rating),
                int(bool(r.in_stock)),
                lookup[r.category],
            )
            for r in df.itertuples(index=False)
        ]
        conn.executemany(
            """INSERT INTO books
               (title, price_gbp, price_inr, rating, in_stock, category_id)
               VALUES (?, ?, ?, ?, ?, ?);""",
            rows,
        )
        conn.commit()

        n_books = conn.execute("SELECT COUNT(*) FROM books;").fetchone()[0]
        n_cats = conn.execute("SELECT COUNT(*) FROM categories;").fetchone()[0]
        print(f"[db] wrote {db_path}: {n_books} books across {n_cats} categories")
    finally:
        conn.close()


if __name__ == "__main__":
    build()
