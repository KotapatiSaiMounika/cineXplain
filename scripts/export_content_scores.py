"""
export_content_scores.py  --  CineXplain, Member 1 -> Member 3 hand-off

Writes data/content_scores.csv with ONE ROW PER (user, movie):

    userId, movieId, content_score

content_score = cosine similarity between the user's taste vector and the movie
(higher = stronger recommendation). The taste vector is built ONLY from the
user's TRAIN ratings, so the scores are safe to use when evaluating the hybrid
on the hidden TEST ratings.

Which users / which movies?
  * Users: the same random sample the evaluation uses (same --users, --seed),
    among users with at least --min-ratings ratings.
  * Movies: for each user, the top --topk unseen movies (movies already rated
    in TRAIN are excluded). Movies not listed for a user were not in that
    user's top-K; Member 3 should treat them as "no content score".

It also writes data/content_split.csv (userId, movieId, rating, set) so
Member 3 can use the exact same train/test split.

Run from the project root:
    python scripts/export_content_scores.py                       # TF-IDF, 3000 users, top 100
    python scripts/export_content_scores.py --method embeddings
    python scripts/export_content_scores.py --users 3000 --topk 200
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.content_recommender import ContentRecommender, load_movies          # noqa: E402
from evaluation.content_metrics import split_ratings                         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", default=str(ROOT / "data" / "ratings.csv"))
    ap.add_argument("--method", choices=["tfidf", "embeddings"], default="tfidf")
    ap.add_argument("--users", type=int, default=3000)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--min-ratings", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "data" / "content_scores.csv"))
    ap.add_argument("--split-out", default=str(ROOT / "data" / "content_split.csv"))
    args = ap.parse_args()

    ratings = pd.read_csv(args.ratings)
    rec = ContentRecommender(load_movies())
    rec.fit_tfidf() if args.method == "tfidf" else rec.fit_embeddings()
    catalog = set(rec.id_to_idx)

    # Same sampling + split as evaluation/content_metrics.py, so users match the evaluation
    rng = np.random.default_rng(args.seed)
    counts = ratings.groupby("userId").size()
    eligible = counts[counts >= args.min_ratings].index.to_numpy()
    users = rng.choice(eligible, size=min(args.users, len(eligible)), replace=False)
    train, test = split_ratings(ratings[ratings.userId.isin(users)], seed=args.seed)

    rows, skipped = [], 0
    for n, u in enumerate(users, 1):
        scores = rec.get_content_scores(train[u], method=args.method, top_k=args.topk)
        if not scores:                      # user has no known movies in train
            skipped += 1
            continue
        rows.extend((int(u), m, s) for m, s in scores.items())
        if n % 500 == 0:
            print(f"  {n}/{len(users)} users done")

    out = pd.DataFrame(rows, columns=["userId", "movieId", "content_score"])
    out["content_score"] = out["content_score"].round(6)
    out = out.sort_values(["userId", "content_score"], ascending=[True, False])

    # ---- sanity checks (fail loudly instead of sending a bad file) ----
    assert not out.isna().any().any(), "missing values in output"
    assert out["movieId"].isin(catalog).all(), "unknown movieId in output"
    assert not out.duplicated(["userId", "movieId"]).any(), "duplicate user-movie pairs"
    assert np.isfinite(out["content_score"]).all(), "non-finite scores"

    out.to_csv(args.out, index=False)

    split_rows = [(int(u), int(m), float(r), name)
                  for name, part in (("train", train), ("test", test))
                  for u, d in part.items() for m, r in d.items()]
    pd.DataFrame(split_rows, columns=["userId", "movieId", "rating", "set"]).to_csv(args.split_out, index=False)

    print(f"\nMethod: {args.method} | users: {out.userId.nunique()} (skipped {skipped}) | "
          f"rows: {len(out):,}")
    print(f"Score range: {out.content_score.min():.4f} to {out.content_score.max():.4f}")
    print(f"Saved -> {args.out}\nSaved -> {args.split_out}")


if __name__ == "__main__":
    main()