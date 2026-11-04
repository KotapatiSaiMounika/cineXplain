"""
content_metrics.py  --  CineXplain, Member 1 evaluation

Measures how good the content-based recommender is, using MovieLens ratings.

Protocol (leave-some-out, per user)
-----------------------------------
1. Pick N users with enough ratings.
2. Hide 20% of each user's ratings (test), keep 80% (train).
3. Build the user's taste profile from TRAIN ratings only.
4. Recommend Top-K movies the user has not seen in train.
5. A recommendation is a HIT if the user rated it >= 4 in the hidden TEST set.

Metrics
-------
Precision@K : hits / K
Recall@K    : hits / (number of liked test movies)
GenreMatch@K: share of recommendations that contain at least one of the user's
              top-3 favourite genres (a simple "relevance" check)
A random baseline is included, so you can show the model beats chance.

Run from the project root:
    python evaluation/content_metrics.py --k 10 --users 200
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # lets us `import src....` when run as a script

from src.content_recommender import ContentRecommender, load_movies  # noqa: E402

LIKE = 4.0   # rating >= 4 counts as "liked / relevant"


def split_ratings(ratings: pd.DataFrame, test_frac: float = 0.2, seed: int = 42):
    """Per-user random split. Returns two dicts: {user: {movieId: rating}}.
    NOTE: the whole team must agree on ONE split strategy -- share this seed."""
    rng = np.random.default_rng(seed)
    train, test = {}, {}
    for uid, grp in ratings.groupby("userId"):
        grp = grp.sample(frac=1.0, random_state=int(rng.integers(1_000_000_000)))
        n_test = max(1, int(round(len(grp) * test_frac)))
        test_part, train_part = grp.iloc[:n_test], grp.iloc[n_test:]
        train[uid] = dict(zip(train_part.movieId.astype(int), train_part.rating))
        test[uid] = dict(zip(test_part.movieId.astype(int), test_part.rating))
    return train, test


def evaluate(rec: ContentRecommender, ratings: pd.DataFrame, method: str,
             k: int = 10, n_users: int = 200, min_ratings: int = 20, seed: int = 42) -> dict:
    """method: 'tfidf' | 'embeddings' | 'random'."""
    rng = np.random.default_rng(seed)
    counts = ratings.groupby("userId").size()
    eligible = counts[counts >= min_ratings].index.to_numpy()
    users = rng.choice(eligible, size=min(n_users, len(eligible)), replace=False)
    train, test = split_ratings(ratings[ratings.userId.isin(users)], seed=seed)

    genre_of = {int(m): set(g.split("|")) - {""}
                for m, g in zip(rec.movies.movieId, rec.movies.genres)}
    catalog = set(rec.id_to_idx)

    P, R, G = [], [], []
    for u in users:
        tr, te = train[u], test[u]
        relevant = {m for m, r in te.items() if r >= LIKE and m in catalog}
        liked_train = [m for m, r in tr.items() if r >= LIKE and m in catalog]
        if not relevant or not liked_train:
            continue

        if method == "random":
            pool = np.array(sorted(catalog - set(tr)))
            recs = [int(x) for x in rng.choice(pool, size=k, replace=False)]
        else:
            recs = list(rec.get_content_scores(tr, method=method, top_k=k))

        hits = len(set(recs) & relevant)
        P.append(hits / k)
        R.append(hits / len(relevant))

        fav = {g for g, _ in Counter(g for m in liked_train for g in genre_of[m]).most_common(3)}
        G.append(np.mean([bool(genre_of[m] & fav) for m in recs]) if recs else 0.0)

    return {"method": method, f"Precision@{k}": np.mean(P), f"Recall@{k}": np.mean(R),
            f"GenreMatch@{k}": np.mean(G), "users_evaluated": len(P)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", default=str(ROOT / "data" / "ratings.csv"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--users", type=int, default=200)
    ap.add_argument("--methods", default="tfidf,embeddings")
    args = ap.parse_args()

    movies = load_movies()
    ratings = pd.read_csv(args.ratings)
    rec = ContentRecommender(movies)

    rows = [evaluate(rec, ratings, "random", args.k, args.users)]
    for m in args.methods.split(","):
        try:
            rec.fit_tfidf() if m == "tfidf" else rec.fit_embeddings()
        except ImportError:
            print("sentence-transformers not installed -> skipping embeddings")
            continue
        rows.append(evaluate(rec, ratings, m, args.k, args.users))

    table = pd.DataFrame(rows).round(4)
    print("\n" + table.to_string(index=False))
    out = ROOT / "evaluation" / "results_content.csv"
    table.to_csv(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()