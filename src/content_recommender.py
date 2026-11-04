"""
content_recommender.py  --  CineXplain, Member 1 (Content-Based Recommendation)

What this file does
-------------------
1. Loads and cleans the movie metadata table.
2. Turns every movie into one text "document" (title + genres + cast + year + plot).
3. Represents each movie as a vector in TWO ways:
      a) TF-IDF            (word counts, re-weighted by how rare a word is)
      b) Sentence embeddings (a neural model that encodes meaning)
4. Uses cosine similarity to find movies close to a movie / a user profile.
5. Exposes get_content_scores()  <-- the function Member 3 (Hybrid) will call.

Run a quick demo from the project root:
    python src/content_recommender.py --movie "Toy Story" --method both --k 10
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------------
# Paths (relative to the project root, so the code works from any folder)
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOVIES = ROOT / "data" / "movies.csv"
DEFAULT_ENRICHED = ROOT / "data" / "movies_enriched.csv"   # optional (plot, cast)
DEFAULT_CACHE = ROOT / "data" / "cache"

# A "profile" is either ONE movieId (int) or a dict {movieId: rating}.
Profile = Union[int, Dict[int, float]]

YEAR_RE = re.compile(r"\((\d{4})\)\s*$")   # matches "(1995)" at the END of a title


# ============================================================================
# PART 1 -- Cleaning helpers
# ============================================================================
def extract_year(title: str) -> float:
    """'Toy Story (1995)' -> 1995.  Returns NaN if there is no year."""
    m = YEAR_RE.search(str(title))
    return float(m.group(1)) if m else np.nan


def strip_year(title: str) -> str:
    """'Toy Story (1995)' -> 'Toy Story'."""
    return YEAR_RE.sub("", str(title)).strip()


def clean_text(text) -> str:
    """Lowercase, keep only letters/digits, collapse spaces. Non-strings -> ''."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def genre_tokens(genres: str) -> list:
    """'Sci-Fi|Action' -> ['scifi', 'action']  (one token per genre)."""
    return [g.lower().replace("-", "").replace(" ", "") for g in str(genres).split("|") if g]


def cast_tokens(cast: str, top_n: int = 5) -> list:
    """'Tom Hanks|Tim Allen' -> ['tomhanks', 'timallen'].
    Names are glued into ONE token so 'Tom' in two different actors doesn't match."""
    names = [n.strip() for n in str(cast).split("|") if n.strip()][:top_n]
    tokens = [re.sub(r"[^a-z]", "", n.lower()) for n in names]
    return [t for t in tokens if t]


def decade_token(year) -> str:
    """1995 -> 'decade1990' so movies from the same decade look a bit similar."""
    return f"decade{int(year) // 10 * 10}" if pd.notna(year) else ""


def load_movies(movies_path=DEFAULT_MOVIES, enriched_path=DEFAULT_ENRICHED) -> pd.DataFrame:
    """
    Load movies.csv (MovieLens: movieId,title,genres) and, if available,
    merge data/movies_enriched.csv (movieId,overview,cast,collection,...) made by
    scripts/enrich_metadata.py.  Works with or without the enriched file.
    """
    movies = pd.read_csv(movies_path)
    movies = movies.drop_duplicates("movieId").copy()
    movies["movieId"] = movies["movieId"].astype(int)
    movies["year"] = movies["title"].map(extract_year)
    movies["clean_title"] = movies["title"].map(strip_year)
    movies["genres"] = movies["genres"].fillna("").replace("(no genres listed)", "")

    if enriched_path and Path(enriched_path).exists():
        extra = pd.read_csv(enriched_path)
        keep = [c for c in ["movieId", "overview", "cast", "collection"] if c in extra.columns]
        extra = extra[keep].drop_duplicates("movieId")
        movies = movies.merge(extra, on="movieId", how="left")

    for col in ["overview", "cast", "collection"]:
        if col not in movies.columns:
            movies[col] = ""
        movies[col] = movies[col].fillna("")
    return movies.reset_index(drop=True)


# ============================================================================
# PART 2 -- The recommender class
# ============================================================================
class ContentRecommender:
    def __init__(self, movies: pd.DataFrame, cache_dir=DEFAULT_CACHE,
                 genre_weight: int = 2, cast_top_n: int = 5):
        self.movies = movies.reset_index(drop=True)
        self.movie_ids = self.movies["movieId"].to_numpy()
        # movieId -> row number (the row number is what indexes the matrices)
        self.id_to_idx = {int(m): i for i, m in enumerate(self.movie_ids)}
        self.cache_dir = Path(cache_dir)
        self.genre_weight = genre_weight
        self.cast_top_n = cast_top_n

        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None       # sparse (n_movies x n_words)
        self.embeddings = None         # dense  (n_movies x 384)

        self.tfidf_texts = self._build_tfidf_texts()
        self.embedding_texts = self._build_embedding_texts()

    # ---------------- text building ----------------
    def _build_tfidf_texts(self) -> list:
        """Cleaned bag-of-words text. Genres are repeated `genre_weight` times so
        they count more; cast names are single tokens; plot words are cleaned."""
        texts = []
        for r in self.movies.itertuples():
            parts = genre_tokens(r.genres) * self.genre_weight
            parts += cast_tokens(r.cast, self.cast_top_n)
            parts.append(decade_token(r.year))
            parts.append(clean_text(r.clean_title))
            parts.append(clean_text(r.overview))
            texts.append(" ".join(p for p in parts if p))
        return texts

    def _build_embedding_texts(self) -> list:
        """Natural sentence-like text. Embedding models understand normal English,
        so we do NOT glue names or repeat genres here."""
        texts = []
        for r in self.movies.itertuples():
            s = f"{r.clean_title}."
            if r.genres:
                s += f" Genres: {r.genres.replace('|', ', ')}."
            cast = [n.strip() for n in str(r.cast).split("|") if n.strip()][: self.cast_top_n]
            if cast:
                s += f" Cast: {', '.join(cast)}."
            if r.overview:
                s += f" Plot: {r.overview}"
            texts.append(s)
        return texts

    # ---------------- fitting ----------------
    def fit_tfidf(self, ngram_range=(1, 1), min_df=1, sublinear_tf=True):
        """
        TF-IDF = Term Frequency x Inverse Document Frequency.
        A word that appears often in ONE movie but in few movies overall gets a
        high weight. stop_words='english' removes 'the', 'and', ...
        Each row is L2-normalised, so dot product == cosine similarity.
        """
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=ngram_range,
            min_df=min_df, sublinear_tf=sublinear_tf,
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.tfidf_texts)
        return self

    def fit_embeddings(self, model_name="all-MiniLM-L6-v2", batch_size=64, use_cache=True):
        """
        Encode every movie text with a Sentence-Transformers model into a
        384-dim vector. Slow the first time (minutes on CPU for ~10k movies),
        so the result is cached in data/cache/ and re-used afterwards.
        """
        digest = hashlib.md5(("||".join(self.embedding_texts) + model_name).encode()).hexdigest()[:10]
        cache_file = self.cache_dir / f"emb_{digest}.npy"
        if use_cache and cache_file.exists():
            self.embeddings = np.load(cache_file)
            print(f"[embeddings] loaded from cache: {cache_file.name}")
            return self

        from sentence_transformers import SentenceTransformer  # imported lazily
        print(f"[embeddings] encoding {len(self.embedding_texts)} movies with {model_name} ...")
        model = SentenceTransformer(model_name)
        self.embeddings = model.encode(
            self.embedding_texts, batch_size=batch_size, show_progress_bar=True,
            normalize_embeddings=True, convert_to_numpy=True,
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, self.embeddings)
        return self

    # ---------------- internals ----------------
    def _matrix(self, method: str):
        if method == "tfidf":
            if self.tfidf_matrix is None:
                raise RuntimeError("Call fit_tfidf() first.")
            return self.tfidf_matrix
        if method == "embeddings":
            if self.embeddings is None:
                raise RuntimeError("Call fit_embeddings() first.")
            return self.embeddings
        raise ValueError("method must be 'tfidf' or 'embeddings'")

    def _parse_profile(self, profile: Profile, like_threshold: float) -> Tuple[Dict[int, float], set]:
        """
        Returns (weights, seen).
          weights: which movies define the user's taste and how strongly.
          seen:    every movie the user already rated (excluded from results).
        A single movieId -> taste is just that movie.
        A {movieId: rating} dict -> taste = movies rated >= like_threshold,
        weighted by rating (fallback: all rated movies if none are liked).
        """
        if isinstance(profile, (int, np.integer)):
            mid = int(profile)
            if mid not in self.id_to_idx:
                raise KeyError(f"movieId {mid} not found in the movie table")
            return {mid: 1.0}, {mid}

        known = {int(m): float(r) for m, r in profile.items() if int(m) in self.id_to_idx}
        liked = {m: r for m, r in known.items() if r >= like_threshold}
        return (liked if liked else known), set(known)

    # ---------------- PUBLIC API (used by Member 3) ----------------
    def get_content_scores(self, profile: Profile, method: str = "tfidf",
                           top_k: Optional[int] = None, exclude_seen: bool = True,
                           like_threshold: float = 4.0) -> Dict[int, float]:
        """
        Agreed team interface:  returns {movieId: score}, best first.
        score = cosine similarity between the user's taste vector and the movie.
        top_k=None returns scores for ALL candidate movies (needed by the hybrid).
        An empty/unknown profile returns {} (Member 3's cold-start gate handles it).
        """
        weights, seen = self._parse_profile(profile, like_threshold)
        if not weights:
            return {}

        X = self._matrix(method)
        idx = np.array([self.id_to_idx[m] for m in weights])
        w = np.array(list(weights.values()), dtype=float)
        w = w / w.sum()                                   # weighted average
        # taste vector = weighted average of the liked movies' vectors
        profile_vec = np.asarray(X[idx].T.dot(w)).reshape(1, -1)
        sims = cosine_similarity(profile_vec, X).ravel()  # one similarity per movie

        mask = np.ones(len(sims), dtype=bool)
        if exclude_seen:
            mask[[self.id_to_idx[m] for m in seen]] = False
        cand = np.where(mask)[0]
        cand = cand[np.argsort(-sims[cand])]
        if top_k:
            cand = cand[:top_k]
        return {int(self.movie_ids[i]): float(sims[i]) for i in cand}

    def recommend(self, profile: Profile, method: str = "tfidf", k: int = 10, **kw) -> pd.DataFrame:
        """Top-K as a readable table: movieId, title, genres, score."""
        scores = self.get_content_scores(profile, method=method, top_k=k, **kw)
        if not scores:
            return pd.DataFrame(columns=["movieId", "title", "genres", "score"])
        df = self.movies.set_index("movieId").loc[list(scores), ["title", "genres"]].reset_index()
        df["score"] = list(scores.values())
        return df

    def compare_methods(self, profile: Profile, k: int = 10) -> dict:
        """TF-IDF vs embeddings: both Top-K lists + how much they overlap (Jaccard)."""
        a = self.recommend(profile, "tfidf", k)
        b = self.recommend(profile, "embeddings", k)
        sa, sb = set(a.movieId), set(b.movieId)
        jac = len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0
        return {"tfidf": a, "embeddings": b, "overlap_jaccard": jac}

    def search_title(self, query: str) -> pd.DataFrame:
        """Case-insensitive title search -> to find a movieId by name."""
        hit = self.movies["title"].str.contains(query, case=False, regex=False, na=False)
        return self.movies.loc[hit, ["movieId", "title", "genres"]]


# ============================================================================
# PART 3 -- Command-line demo
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="CineXplain content-based demo")
    ap.add_argument("--movie", required=True, help='part of a title, e.g. "Toy Story"')
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--method", choices=["tfidf", "embeddings", "both"], default="tfidf")
    args = ap.parse_args()

    movies = load_movies()
    rec = ContentRecommender(movies)
    matches = rec.search_title(args.movie)
    if matches.empty:
        print(f"No movie title contains '{args.movie}'.")
        return
    movie_id = int(matches.iloc[0].movieId)
    print(f"\nSeed movie: {matches.iloc[0].title}  (movieId={movie_id})")
    if len(matches) > 1:
        print(f"({len(matches)} titles matched; using the first.)")

    methods = ["tfidf", "embeddings"] if args.method == "both" else [args.method]
    for m in methods:
        rec.fit_tfidf() if m == "tfidf" else rec.fit_embeddings()
        print(f"\n--- Top {args.k} by {m.upper()} ---")
        print(rec.recommend(movie_id, m, args.k).to_string(index=False))

    if args.method == "both":
        print(f"\nTop-{args.k} overlap (Jaccard): {rec.compare_methods(movie_id, args.k)['overlap_jaccard']:.2f}")


if __name__ == "__main__":
    main()