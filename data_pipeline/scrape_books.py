"""
Module 1, step 1: scrape -> clean -> currency-convert.

Scrapes books.toscrape.com (a public scraping-practice site: no login, no API
key, no paid tier), cleans the raw text fields into proper types, converts GBP
to INR using the project's fixed baseline rate, and writes books_clean.csv.

Run:  python scrape_books.py
"""

from __future__ import annotations

import re
import sys
import time
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"

# Project-defined fixed baseline conversion rate for this assignment.
# This is an artificial constant, NOT a live or historical market rate, so it
# needs no lookup, no API call and no date reference.
GBP_TO_INR = 105.50

# We scrape these categories (>= 3 required). Together they comfortably exceed
# the required minimum of 60 books.
CATEGORIES = [
    "Fiction",
    "Mystery",
    "Historical Fiction",
    "Romance",
    "Nonfiction",
]

RATING_WORDS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

HEADERS = {"User-Agent": "zepto-capstone-scraper/1.0 (educational use)"}


def get_soup(url: str) -> BeautifulSoup:
    """GET a URL and return parsed HTML. Raises on non-200 responses."""
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    # books.toscrape.com serves UTF-8 but does not always declare it correctly.
    response.encoding = response.apparent_encoding or "utf-8"
    return BeautifulSoup(response.text, "html.parser")


def get_category_links() -> dict[str, str]:
    """Map category name -> absolute listing URL from the site's sidebar."""
    soup = get_soup(BASE_URL)
    links: dict[str, str] = {}
    for anchor in soup.select("div.side_categories ul li ul li a"):
        name = anchor.get_text(strip=True)
        links[name] = urljoin(BASE_URL, anchor["href"])
    return links


def parse_listing_page(soup: BeautifulSoup, category: str) -> list[dict]:
    """Extract one row per book card on a listing page (raw text, no cleaning)."""
    rows = []
    for card in soup.select("article.product_pod"):
        title_tag = card.select_one("h3 a")
        price_tag = card.select_one("p.price_color")
        rating_tag = card.select_one("p.star-rating")
        availability_tag = card.select_one("p.instock.availability")

        # star-rating carries its value as the second CSS class, e.g.
        # <p class="star-rating Three">
        rating_text = ""
        if rating_tag is not None:
            classes = [c for c in rating_tag.get("class", []) if c != "star-rating"]
            rating_text = classes[0] if classes else ""

        rows.append(
            {
                "title": title_tag["title"].strip() if title_tag else "",
                "price": price_tag.get_text(strip=True) if price_tag else "",
                "star_rating": rating_text,
                "availability": (
                    availability_tag.get_text(strip=True) if availability_tag else ""
                ),
                "category": category,
            }
        )
    return rows


def scrape_category(name: str, url: str, polite_delay: float = 0.3) -> list[dict]:
    """Scrape every paginated page of a single category listing."""
    rows: list[dict] = []
    next_url = url
    while next_url:
        soup = get_soup(next_url)
        rows.extend(parse_listing_page(soup, name))
        next_link = soup.select_one("li.next a")
        next_url = urljoin(next_url, next_link["href"]) if next_link else None
        time.sleep(polite_delay)  # be polite to the practice server
    return rows


def scrape_all() -> pd.DataFrame:
    available = get_category_links()
    all_rows: list[dict] = []
    for category in CATEGORIES:
        if category not in available:
            print(f"[warn] category '{category}' not found on site, skipping")
            continue
        rows = scrape_category(category, available[category])
        print(f"[scrape] {category:<20} {len(rows):>4} books")
        all_rows.extend(rows)
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def parse_price(raw: str) -> float | None:
    """'Â£51.77' / '£51.77' -> 51.77 ; unparseable -> None."""
    if not isinstance(raw, str):
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", raw.replace(",", ""))
    return float(match.group(1)) if match else None


def parse_rating(raw: str) -> int | None:
    """'Three' -> 3 ; unknown word -> None."""
    return RATING_WORDS.get(str(raw).strip().capitalize())


def parse_in_stock(raw: str) -> bool | None:
    """'In stock (22 available)' -> True ; 'Out of stock' -> False."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.lower()
    if "in stock" in text:
        return True
    if "out of stock" in text or "unavailable" in text:
        return False
    return None


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning / bad-row policy (documented in the README):

      * price_gbp : numeric  -> MEDIAN IMPUTATION on parse failure. Price is the
        column every downstream aggregation depends on, and the catalogue's
        price distribution is tight, so a median stand-in keeps the row usable
        without distorting per-category averages.
      * rating    : numeric  -> MEDIAN IMPUTATION on parse failure, same reasoning.
      * in_stock  : boolean  -> NOT imputable (there is no meaningful "median"
        availability, and guessing it would invent stock state), so rows whose
        availability text cannot be parsed are DROPPED.
      * title     -> a row with no title is not a usable catalogue record: DROPPED.

    The pipeline therefore never crashes on messy rows; it either repairs them
    or removes them, and it reports exactly what it did.
    """
    out = df.copy()

    out["price_gbp"] = out["price"].apply(parse_price)
    out["rating"] = out["star_rating"].apply(parse_rating)
    out["in_stock"] = out["availability"].apply(parse_in_stock)

    report = {
        "rows_scraped": len(out),
        "price_parse_failures": int(out["price_gbp"].isna().sum()),
        "rating_parse_failures": int(out["rating"].isna().sum()),
        "availability_parse_failures": int(out["in_stock"].isna().sum()),
        "blank_titles": int((out["title"].fillna("").str.strip() == "").sum()),
    }

    # --- numeric fields: median imputation -----------------------------------
    if out["price_gbp"].isna().any():
        out["price_gbp"] = out["price_gbp"].fillna(out["price_gbp"].median())
    if out["rating"].isna().any():
        out["rating"] = out["rating"].fillna(out["rating"].median())

    # --- non-imputable fields: drop ------------------------------------------
    out = out[out["title"].fillna("").str.strip() != ""]
    out = out.dropna(subset=["in_stock"])

    # --- final typing --------------------------------------------------------
    out["price_gbp"] = out["price_gbp"].astype(float).round(2)
    out["rating"] = out["rating"].astype(int)
    out["in_stock"] = out["in_stock"].astype(bool)

    # --- fixed-rate currency conversion (required baseline, no network) ------
    out["price_inr"] = (out["price_gbp"] * GBP_TO_INR).round(2)

    report["rows_kept"] = len(out)
    print("\n[clean] " + " | ".join(f"{k}={v}" for k, v in report.items()))

    return out[
        ["title", "category", "price_gbp", "price_inr", "rating", "in_stock"]
    ].reset_index(drop=True)


def main() -> int:
    try:
        raw = scrape_all()
    except requests.RequestException as exc:
        print(f"[error] scraping failed: {exc}", file=sys.stderr)
        print("        books.toscrape.com must be reachable to run this step.",
              file=sys.stderr)
        return 1

    if raw.empty:
        print("[error] no rows scraped", file=sys.stderr)
        return 1

    cleaned = clean(raw)

    assert len(cleaned) >= 60, f"expected >= 60 books, got {len(cleaned)}"
    assert cleaned["category"].nunique() >= 3, "expected >= 3 categories"

    raw.to_csv("books_raw.csv", index=False)
    cleaned.to_csv("books_clean.csv", index=False)

    print(f"\n[done] {len(cleaned)} books across "
          f"{cleaned['category'].nunique()} categories -> books_clean.csv")
    print(f"[done] conversion used fixed baseline rate 1 GBP = {GBP_TO_INR} INR")
    print(cleaned.dtypes)
    print(cleaned.head())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
