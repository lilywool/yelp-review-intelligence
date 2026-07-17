"""
Reproducible sample extraction from the raw Yelp Open Dataset.

Bridges the gap between Yelp's raw JSON download and the CSVs that
real_feature_engineering.py consumes. Anyone with the Yelp dataset and this
script can regenerate byte-identical input samples (fixed seed), which is what
makes this repo fully reproducible without redistributing Yelp's data.

Input (from https://business.yelp.com/data/resources/open-dataset/):
    yelp_academic_dataset_business.json
    yelp_academic_dataset_review.json
    yelp_academic_dataset_user.json   (optional - adds user columns if present)

Output (per requested industry):
    {label}_sample_{n}.csv  with columns:
        review_id, user_id, business_id, stars, raw_review, review_date,
        year_month, name, city, state, business_review_count, is_open,
        primary_industry
        [+ user_name, user_stars_avg, user_review_count if user file provided]

Usage:
    python make_samples.py --json-dir path/to/Yelp-JSON \
        --industry "Mexican:mexican" "Hair Salons:hair" \
        --n 15000 --seed 222 --outdir ../data/raw

    --industry takes CATEGORY:LABEL pairs: CATEGORY is matched (case-insensitive,
    substring) against each business's Yelp `categories` string; LABEL names the
    output file. Repeat the flag value for multiple industries.

Design notes:
    - Two streaming passes over review JSON, nothing large held in memory:
      pass 1 counts matching reviews per industry; pass 2 collects the rows
      chosen by the seeded RNG. Runs fine on a laptop against the full 8.6M.
    - Sampling is uniform-random over each industry's matching reviews with
      numpy's default_rng(seed), so identical inputs + identical seed =
      identical output, regardless of machine.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_business_index(business_path: Path, category_queries: dict) -> dict:
    """Map business_id -> business info + matched industry label.

    category_queries: {query_string_lowercase: label}
    A business matches the FIRST query found in its categories string.
    """
    index = {}
    with open(business_path, "r", encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            cats = (b.get("categories") or "").lower()
            for query, label in category_queries.items():
                if query in cats:
                    index[b["business_id"]] = {
                        "name": b.get("name"),
                        "city": b.get("city"),
                        "state": b.get("state"),
                        "business_review_count": b.get("review_count"),
                        "is_open": b.get("is_open"),
                        "primary_industry": label,
                    }
                    break
    return index


def load_user_index(user_path: Path) -> dict:
    """Map user_id -> user info. Optional; returns {} if path is None."""
    if user_path is None:
        return {}
    index = {}
    with open(user_path, "r", encoding="utf-8") as f:
        for line in f:
            u = json.loads(line)
            index[u["user_id"]] = {
                "user_name": u.get("name"),
                "user_stars_avg": u.get("average_stars"),
                "user_review_count": u.get("review_count"),
            }
    return index


def count_matching_reviews(review_path: Path, business_index: dict) -> dict:
    """Pass 1: count reviews per industry label (streaming, low memory)."""
    counts = {}
    with open(review_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            info = business_index.get(r["business_id"])
            if info:
                label = info["primary_industry"]
                counts[label] = counts.get(label, 0) + 1
    return counts


def collect_samples(review_path: Path, business_index: dict, user_index: dict,
                    chosen: dict) -> dict:
    """Pass 2: stream reviews again, keep the pre-chosen indices per label."""
    seen = {label: 0 for label in chosen}
    rows = {label: [] for label in chosen}
    with open(review_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            info = business_index.get(r["business_id"])
            if not info:
                continue
            label = info["primary_industry"]
            if seen[label] in chosen[label]:
                date = r.get("date", "")
                row = {
                    "review_id": r["review_id"],
                    "user_id": r["user_id"],
                    "business_id": r["business_id"],
                    "stars": r["stars"],
                    "raw_review": r["text"],
                    "review_date": date,
                    "year_month": date[:7] if date else "",
                    **info,
                    **user_index.get(r["user_id"], {}),
                }
                rows[label].append(row)
            seen[label] += 1
    return rows


def main():
    parser = argparse.ArgumentParser(description="Reproducible Yelp industry sampler.")
    parser.add_argument("--json-dir", required=True,
                        help="Folder containing the yelp_academic_dataset_*.json files.")
    parser.add_argument("--industry", nargs="+", required=True,
                        help='CATEGORY:LABEL pairs, e.g. "Mexican:mexican" "Hair Salons:hair"')
    parser.add_argument("--n", type=int, default=15000, help="Reviews per industry (default 15000).")
    parser.add_argument("--seed", type=int, default=222, help="RNG seed (default 222).")
    parser.add_argument("--outdir", default=".", help="Output folder for sample CSVs.")
    parser.add_argument("--no-users", action="store_true",
                        help="Skip the user JSON (faster; omits user_* columns).")
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    business_path = json_dir / "yelp_academic_dataset_business.json"
    review_path = json_dir / "yelp_academic_dataset_review.json"
    user_path = None if args.no_users else json_dir / "yelp_academic_dataset_user.json"

    for p in [business_path, review_path] + ([user_path] if user_path else []):
        if not p.exists():
            raise SystemExit(f"Missing expected file: {p}")

    category_queries = {}
    for pair in args.industry:
        if ":" not in pair:
            raise SystemExit(f'--industry entries must be CATEGORY:LABEL, got "{pair}"')
        query, label = pair.split(":", 1)
        category_queries[query.strip().lower()] = label.strip()

    print("Indexing businesses...")
    business_index = load_business_index(business_path, category_queries)
    print(f"  {len(business_index):,} businesses matched {list(category_queries.values())}")

    user_index = {}
    if user_path:
        print("Indexing users (this is the slow part; use --no-users to skip)...")
        user_index = load_user_index(user_path)
        print(f"  {len(user_index):,} users indexed")

    print("Pass 1/2: counting matching reviews...")
    counts = count_matching_reviews(review_path, business_index)
    for label, c in counts.items():
        print(f"  {label}: {c:,} reviews available")

    rng = np.random.default_rng(args.seed)
    chosen = {}
    for label in category_queries.values():
        available = counts.get(label, 0)
        take = min(args.n, available)
        if take < args.n:
            print(f"  WARNING: {label} has only {available:,} reviews; sampling all of them.")
        chosen[label] = set(rng.choice(available, size=take, replace=False).tolist())

    print("Pass 2/2: collecting sampled reviews...")
    rows = collect_samples(review_path, business_index, user_index, chosen)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for label, records in rows.items():
        df = pd.DataFrame(records)
        outfile = outdir / f"{label}_sample_{len(df)}.csv"
        df.to_csv(outfile, index=False)
        print(f"Saved {len(df):,} rows -> {outfile}")

    print(f"\nDone. Seed={args.seed}; identical inputs + seed reproduce identical samples.")


if __name__ == "__main__":
    main()
