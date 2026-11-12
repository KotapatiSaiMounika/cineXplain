"""Streamlit-facing recommendation pipeline for CineXplain.

This module connects the project's existing content, hybrid and MMR ideas in a
small, predictable pipeline suitable for an interactive demo. It deliberately
uses a lightweight item-item collaborative model instead of training the
large offline MovieLens KNN/SVD experiment every time Streamlit starts.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from sklearn.metrics.pairwise import cosine_similarity

from src.content_recommender import ContentRecommender
from src.constraints import MovieConstraintChecker
from src.hybrid_recommender import calculate_alpha, hybrid_recommender
from src.mmr import mmr_rerank


_REQUIRED_MOVIE_COLUMNS = {"movieId", "title"}
_REQUIRED_RATING_COLUMNS = {"userId", "movieId", "rating"}
_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def _rename_aliases(frame: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    """Rename common column aliases without overwriting canonical columns."""
    rename_map: dict[str, str] = {}
    existing = set(frame.columns)
    for source, target in aliases.items():
        if source in existing and target not in existing:
            rename_map[source] = target
    return frame.rename(columns=rename_map)


def normalise_movies(
    movies: pd.DataFrame,
    enriched: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Validate MovieLens-style movie data and standardise optional metadata."""
    if movies is None or movies.empty:
        raise ValueError("The movie dataset is empty. Upload a valid movies.csv file.")

    result = movies.copy()
    result.columns = [str(column).strip() for column in result.columns]
    result = _rename_aliases(
        result,
        {"movie_id": "movieId", "movieID": "movieId", "name": "title", "movie_title": "title"},
    )

    missing = _REQUIRED_MOVIE_COLUMNS - set(result.columns)
    if missing:
        raise ValueError(f"movies.csv is missing required columns: {sorted(missing)}")

    result["movieId"] = pd.to_numeric(result["movieId"], errors="coerce")
    result = result.dropna(subset=["movieId", "title"]).copy()
    result["movieId"] = result["movieId"].astype("int64")
    result["title"] = result["title"].astype(str).str.strip()
    result = result[result["title"].ne("")].drop_duplicates("movieId", keep="first")

    if "genres" not in result.columns:
        result["genres"] = ""
    result["genres"] = result["genres"].fillna("").astype(str)
    result["genres"] = result["genres"].replace("(no genres listed)", "")

    # Merge enrichment data only for metadata columns that are not already present.
    if enriched is not None and not enriched.empty:
        extra = enriched.copy()
        extra.columns = [str(column).strip() for column in extra.columns]
        extra = _rename_aliases(extra, {"movie_id": "movieId", "movieID": "movieId"})
        if "movieId" in extra.columns:
            optional = [
                column
                for column in [
                    "overview", "plot", "cast", "collection", "director",
                    "keywords", "tagline", "poster_url", "poster_path",
                ]
                if column in extra.columns
            ]
            if optional:
                extra = extra[["movieId", *optional]].drop_duplicates("movieId")
                # Keep values already present in the main file, but fill blanks
                # from the optional enrichment file.
                conflicts = {column: f"{column}__enriched" for column in optional if column in result.columns}
                extra = extra.rename(columns=conflicts)
                result = result.merge(extra, on="movieId", how="left")
                for column, enriched_column in conflicts.items():
                    current = result[column].fillna("").astype(str)
                    result[column] = current.where(current.str.strip().ne(""), result[enriched_column])
                    result = result.drop(columns=[enriched_column])

    for column in [
        "overview", "plot", "cast", "collection", "director", "keywords",
        "tagline", "poster_url", "poster_path",
    ]:
        if column not in result.columns:
            result[column] = ""
        result[column] = result[column].fillna("").astype(str)

    extracted_year = result["title"].str.extract(_YEAR_RE)[0]
    if "year" in result.columns:
        year_values = pd.to_numeric(result["year"], errors="coerce")
    elif "release_year" in result.columns:
        year_values = pd.to_numeric(result["release_year"], errors="coerce")
    else:
        year_values = pd.Series(np.nan, index=result.index, dtype=float)
    result["year"] = year_values.fillna(pd.to_numeric(extracted_year, errors="coerce"))
    result["year"] = result["year"].fillna(0).astype(int)
    if "plot" in result.columns:
        overview = result["overview"].fillna("").astype(str)
        result["overview"] = overview.where(overview.str.strip().ne(""), result["plot"].fillna(""))
    result["clean_title"] = result["title"].str.replace(_YEAR_RE, "", regex=True).str.strip()

    # Keep known columns first, but preserve any additional uploaded metadata.
    preferred = [
        "movieId", "title", "clean_title", "year", "genres", "overview", "plot",
        "cast", "director", "keywords", "tagline", "collection", "poster_url", "poster_path",
    ]
    ordered = [column for column in preferred if column in result.columns]
    ordered += [column for column in result.columns if column not in ordered]
    result = result[ordered].sort_values("title").reset_index(drop=True)

    if result.empty:
        raise ValueError("No valid movies remained after cleaning the uploaded dataset.")
    return result


def normalise_ratings(ratings: pd.DataFrame | None, movie_ids: set[int]) -> pd.DataFrame:
    """Validate and reduce ratings to one numeric user/movie rating per pair."""
    columns = ["userId", "movieId", "rating"]
    if ratings is None or ratings.empty:
        return pd.DataFrame(columns=columns)

    result = ratings.copy()
    result.columns = [str(column).strip() for column in result.columns]
    result = _rename_aliases(
        result,
        {"user_id": "userId", "movie_id": "movieId", "movieID": "movieId", "stars": "rating"},
    )
    missing = _REQUIRED_RATING_COLUMNS - set(result.columns)
    if missing:
        raise ValueError(f"ratings.csv is missing required columns: {sorted(missing)}")

    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=columns).copy()
    result = result[result["movieId"].isin(movie_ids)]
    result = result[result["rating"].between(0.5, 5.0)]
    if result.empty:
        return pd.DataFrame(columns=columns)

    result["userId"] = result["userId"].astype("int64")
    result["movieId"] = result["movieId"].astype("int64")
    result = result.groupby(["userId", "movieId"], as_index=False)["rating"].mean()
    return result[columns]


class CineXplainEngine:
    """Build and run content, collaborative, hybrid and MMR recommendations."""

    def __init__(
        self,
        movies: pd.DataFrame,
        ratings: pd.DataFrame | None = None,
        enriched: pd.DataFrame | None = None,
        cache_dir: str | Path = "data/cache",
    ) -> None:
        self.movies = normalise_movies(movies, enriched=enriched)
        self.ratings = normalise_ratings(
            ratings,
            set(self.movies["movieId"].astype(int).tolist()),
        )
        self.movie_ids = self.movies["movieId"].astype(int).to_numpy()
        self.id_to_row = {int(movie_id): row for row, movie_id in enumerate(self.movie_ids)}
        self.movie_lookup = self.movies.set_index("movieId", drop=False)
        self.constraint_checker = MovieConstraintChecker(self.movies)

        self.content_model = ContentRecommender(self.movies, cache_dir=cache_dir)
        self.content_model.fit_tfidf(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self._build_collaborative_index()
        self._build_popularity()

    def _build_collaborative_index(self) -> None:
        """Create a user-centred item-user matrix for lightweight item-item KNN."""
        self.item_user = None
        self.item_user_normalised = None
        self.item_user_binary = None
        self.user_count = 0
        if self.ratings.empty:
            return

        user_ids = np.sort(self.ratings["userId"].unique())
        user_to_col = {int(user_id): col for col, user_id in enumerate(user_ids)}
        rows = self.ratings["movieId"].map(self.id_to_row).to_numpy(dtype=np.int32)
        cols = self.ratings["userId"].map(user_to_col).to_numpy(dtype=np.int32)
        user_means = self.ratings.groupby("userId")["rating"].mean()
        means = self.ratings["userId"].map(user_means).to_numpy(dtype=np.float32)
        centred = self.ratings["rating"].to_numpy(dtype=np.float32) - means

        shape = (len(self.movie_ids), len(user_ids))
        self.item_user = csr_matrix((centred, (rows, cols)), shape=shape, dtype=np.float32)
        self.item_user_binary = csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=shape,
            dtype=np.float32,
        )
        norms = np.sqrt(np.asarray(self.item_user.multiply(self.item_user).sum(axis=1)).ravel())
        inverse_norms = np.divide(
            1.0,
            norms,
            out=np.zeros_like(norms, dtype=np.float32),
            where=norms > 0,
        )
        self.item_user_normalised = diags(inverse_norms) @ self.item_user
        self.user_count = len(user_ids)

    def _build_popularity(self) -> None:
        if self.ratings.empty:
            self.global_mean = 3.5
            self.popularity = pd.Series(0.0, index=self.movie_ids)
            return
        stats = self.ratings.groupby("movieId")["rating"].agg(["mean", "count"])
        self.global_mean = float(self.ratings["rating"].mean())
        prior_strength = 8.0
        bayes = (
            stats["count"] * stats["mean"] + prior_strength * self.global_mean
        ) / (stats["count"] + prior_strength)
        self.popularity = bayes.reindex(self.movie_ids).fillna(self.global_mean)

    @property
    def has_collaborative_data(self) -> bool:
        return self.item_user_normalised is not None and self.user_count >= 2

    def _popularity_scores(self) -> dict[int, float]:
        if len(self.popularity) == 0:
            return {}
        return {
            int(movie_id): float(score)
            for movie_id, score in self.popularity.items()
        }

    def _collaborative_scores(self, profile: dict[int, float], shrinkage: float = 8.0) -> dict[int, float]:
        """Score movies by similarity to liked/disliked seed movies.

        Similarity is adjusted cosine over user-mean-centred ratings. A small
        co-rating shrinkage term discourages high similarities based on only a
        handful of common raters.
        """
        if not self.has_collaborative_data:
            return {}

        total_scores = np.zeros(len(self.movie_ids), dtype=np.float64)
        total_weight = 0.0
        for movie_id, rating in profile.items():
            row = self.id_to_row.get(int(movie_id))
            if row is None:
                continue
            rating = float(rating)
            if rating >= 4.0:
                signed_weight = rating - 3.0
            elif rating <= 2.0:
                signed_weight = -(3.0 - rating)
            else:
                signed_weight = 0.0
            if signed_weight == 0:
                continue

            seed_vector = self.item_user_normalised.getrow(row)
            if seed_vector.nnz == 0:
                continue
            similarities = (seed_vector @ self.item_user_normalised.T).toarray().ravel()
            overlaps = (self.item_user_binary.getrow(row) @ self.item_user_binary.T).toarray().ravel()
            shrink = overlaps / (overlaps + shrinkage)
            similarities = similarities * shrink
            total_scores += signed_weight * similarities
            total_weight += abs(signed_weight)

        if total_weight == 0:
            return {}

        total_scores /= total_weight
        total_scores = np.maximum(total_scores, 0.0)
        total_scores[[self.id_to_row[mid] for mid in profile if mid in self.id_to_row]] = 0.0
        if not np.any(total_scores > 0):
            return {}

        return {
            int(self.movie_ids[index]): float(score)
            for index, score in enumerate(total_scores)
            if score > 0
        }

    def _explain(self, movie_id: int, profile: dict[int, float], content_score: float, cf_score: float) -> str:
        candidate = self.movie_lookup.loc[movie_id]
        candidate_genres = {
            item.strip().lower()
            for item in str(candidate.get("genres", "")).split("|")
            if item.strip()
        }
        liked_genres: set[str] = set()
        for seed_id, rating in profile.items():
            if rating >= 4.0 and seed_id in self.movie_lookup.index:
                liked_genres.update(
                    item.strip().lower()
                    for item in str(self.movie_lookup.loc[seed_id].get("genres", "")).split("|")
                    if item.strip()
                )
        shared = sorted(candidate_genres & liked_genres)
        pieces = []
        if shared:
            pieces.append("Shares your taste for " + ", ".join(g.title() for g in shared[:3]))
        if content_score >= 0.35:
            pieces.append("strong content match")
        if cf_score >= 0.5:
            pieces.append("supported by similar rating patterns")
        if not pieces:
            pieces.append("balances your profile with variety")
        return " · ".join(pieces)

    def recommend(
        self,
        profile: dict[int, float],
        top_k: int = 10,
        diversity: float = 0.72,
        excluded_genres: list[str] | None = None,
        min_year: int | None = None,
        max_year: int | None = None,
        max_per_franchise: int | None = None,
        candidate_pool: int = 100,
    ) -> dict[str, Any]:
        """Generate baseline hybrid rankings and MMR-diversified results."""
        if not profile:
            raise ValueError("Choose at least one favourite movie to build your taste profile.")
        if not any(float(value) >= 4.0 for value in profile.values()):
            raise ValueError("Choose at least one movie rated 4 or 5 as a favourite.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if not 0.0 <= diversity <= 1.0:
            raise ValueError("diversity must be between 0 and 1.")

        profile = {int(key): float(value) for key, value in profile.items() if int(key) in self.id_to_row}
        if not profile:
            raise ValueError("None of the selected movies are present in the current dataset.")

        content_scores = self.content_model.get_content_scores(
            profile,
            method="tfidf",
            top_k=None,
            exclude_seen=True,
            like_threshold=4.0,
        )
        interaction_count = len(profile)
        if self.ratings.empty:
            cf_scores = {}
            cf_source = "unavailable"
            alpha = 1.0
        else:
            cf_scores = self._collaborative_scores(profile)
            cf_source = "item-item collaborative filtering" if cf_scores else "popularity fallback"
            if not cf_scores:
                cf_scores = self._popularity_scores()
            alpha = calculate_alpha(interaction_count, k=68)
        hybrid = hybrid_recommender(
            content_scores=content_scores,
            cf_scores=cf_scores,
            num_interactions=interaction_count,
            top_k=max(candidate_pool, top_k),
            k=68,
        )
        if hybrid.empty:
            raise ValueError("The recommender could not create candidates from this profile.")

        # Attach catalogue metadata and apply hard user constraints before MMR.
        ranked = hybrid.merge(self.movies, on="movieId", how="left")
        seen = set(profile)
        ranked = ranked[~ranked["movieId"].isin(seen)].copy()
        ranked = ranked.sort_values("hybrid_score", ascending=False).drop_duplicates("movieId")
        allowed = self.constraint_checker.apply_constraints(
            dict(zip(ranked["movieId"].astype(int), ranked["hybrid_score"].astype(float))),
            {
                "excluded_genres": excluded_genres or [],
                "min_year": min_year,
                "max_year": max_year,
                "max_sequels": max_per_franchise,
            },
        )
        ranked = ranked[ranked["movieId"].isin(allowed)].copy()

        if ranked.empty:
            raise ValueError("No movies match those filters. Try including more genres or widening the year range.")

        ranked = ranked.head(min(candidate_pool, len(ranked))).reset_index(drop=True)
        candidate_ids = ranked["movieId"].astype(int).tolist()
        candidate_rows = [self.id_to_row[movie_id] for movie_id in candidate_ids]
        candidate_vectors = self.content_model.tfidf_matrix[candidate_rows]
        similarity = cosine_similarity(candidate_vectors, dense_output=True)
        similarity = np.clip(similarity, 0.0, 1.0)
        np.fill_diagonal(similarity, 1.0)

        relevance = ranked["hybrid_score"].to_numpy(dtype=float)
        mmr_output = mmr_rerank(
            movie_ids=candidate_ids,
            relevance_scores=relevance,
            similarity_matrix=similarity,
            k=min(top_k, len(candidate_ids)),
            lambda_param=float(diversity),
        )
        mmr_frame = pd.DataFrame(
            mmr_output,
            columns=["movieId", "relevance_score", "mmr_score"],
        )
        diverse = mmr_frame.merge(ranked, on="movieId", how="left")
        diverse["why_recommended"] = diverse.apply(
            lambda row: self._explain(
                int(row["movieId"]),
                profile,
                float(row.get("content_score", 0.0)),
                float(row.get("cf_score", 0.0)),
            ),
            axis=1,
        )

        baseline = ranked.head(top_k).copy()
        baseline["why_recommended"] = baseline.apply(
            lambda row: self._explain(
                int(row["movieId"]),
                profile,
                float(row.get("content_score", 0.0)),
                float(row.get("cf_score", 0.0)),
            ),
            axis=1,
        )

        def ild(movie_list: list[int]) -> float:
            if len(movie_list) < 2:
                return 0.0
            positions = {movie_id: index for index, movie_id in enumerate(candidate_ids)}
            values = [
                similarity[positions[left], positions[right]]
                for i, left in enumerate(movie_list)
                for right in movie_list[i + 1 :]
            ]
            return float(1.0 - np.mean(values)) if values else 0.0

        output_columns = [
            "movieId", "title", "year", "genres", "overview", "poster_url", "poster_path",
            "content_score", "cf_score", "hybrid_score", "mmr_score", "why_recommended",
        ]
        for frame in (baseline, diverse):
            for column in output_columns:
                if column not in frame.columns:
                    frame[column] = np.nan

        return {
            "hybrid": baseline[output_columns].reset_index(drop=True),
            "diverse": diverse[output_columns].reset_index(drop=True),
            "alpha": float(alpha),
            "content_weight": float(alpha),
            "collaborative_weight": float(1.0 - alpha),
            "collaborative_source": cf_source,
            "candidate_count": int(len(ranked)),
            "hybrid_ild": ild(baseline["movieId"].astype(int).tolist()),
            "diverse_ild": ild(diverse["movieId"].astype(int).tolist()),
            "lambda": float(diversity),
            "catalogue_size": int(len(self.movies)),
            "ratings_count": int(len(self.ratings)),
            "user_count": int(self.ratings["userId"].nunique()) if not self.ratings.empty else 0,
        }
