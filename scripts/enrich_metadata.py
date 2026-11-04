"""
enrich_metadata.py -- OPTIONAL but recommended.

MovieLens has NO plot or cast. This script uses links.csv (movieId -> tmdbId)
and the free TMDB API to download, for every movie:
    overview (plot), cast (top 8 names), collection (franchise, e.g. "Toy Story Collection")
and writes data/movies_enriched.csv. content_recommender.py picks it up automatically.
(`collection` is also what Member 5 needs to detect sequels.)

Get a free API key: https://www.themoviedb.org/settings/api  (the "API Key (v3 auth)")

Run from the project root:
    export TMDB_API_KEY=your_key            # Windows PowerShell: $env:TMDB_API_KEY="your_key"
    python scripts/enrich_metadata.py --limit 50     # test on 50 movies first
    python scripts/enrich_metadata.py                # then everything (resumable)
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
URL = "https://api.themoviedb.org/3/movie/{}"


def parse_movie(js: dict) -> dict:
    """Pick the fields we need out of TMDB's JSON response."""
    cast = [c.get("name", "") for c in (js.get("credits", {}) or {}).get("cast", [])[:8]]
    coll = (js.get("belongs_to_collection") or {}).get("name", "")
    return {"overview": js.get("overview", "") or "", "cast": "|".join(n for n in cast if n),
            "collection": coll}


def fetch(tmdb_id: int, api_key: str, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(URL.format(tmdb_id),
                             params={"api_key": api_key, "append_to_response": "credits"},
                             timeout=20)
            if r.status_code == 200:
                return parse_movie(r.json())
            if r.status_code == 404:
                break                      # movie missing on TMDB -> leave empty
            if r.status_code == 429:       # rate limited -> wait and retry
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code == 401:
                raise SystemExit("TMDB rejected the API key (401). Check TMDB_API_KEY.")
        except requests.RequestException:
            time.sleep(1 + attempt)
    return {"overview": "", "cast": "", "collection": ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", default=str(ROOT / "data" / "links.csv"))
    ap.add_argument("--out", default=str(ROOT / "data" / "movies_enriched.csv"))
    ap.add_argument("--api-key", default=os.environ.get("TMDB_API_KEY"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if not args.api_key:
        raise SystemExit("Set TMDB_API_KEY or pass --api-key.")

    links = pd.read_csv(args.links).dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype(int)

    out = Path(args.out)
    done = set(pd.read_csv(out).movieId) if out.exists() else set()   # resume support
    todo = links[~links.movieId.isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"{len(done)} already done, {len(todo)} to fetch")

    chunk = 200
    for start in range(0, len(todo), chunk):
        part = todo.iloc[start:start + chunk]
        with ThreadPoolExecutor(args.workers) as ex:
            results = list(ex.map(lambda t: fetch(t, args.api_key), part.tmdbId))
        df = pd.DataFrame(results)
        df.insert(0, "movieId", part.movieId.to_numpy())
        df.to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"  saved {start + len(part)}/{len(todo)}")
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()