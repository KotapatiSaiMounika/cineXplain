
import numpy as np
import pandas as pd


def precision_at_k(recommended, relevant, k=10):
    """Fraction of top-K recommendations that are relevant."""

    if k <= 0:
        raise ValueError("k must be positive")

    top_k = recommended[:k]

    if not top_k:
        return 0.0

    hits = len(set(top_k) & set(relevant))

    return hits / k


def recall_at_k(recommended, relevant, k=10):
    """Fraction of relevant movies found in top-K."""

    if k <= 0:
        raise ValueError("k must be positive")

    if not relevant:
        return 0.0

    top_k = recommended[:k]

    hits = len(set(top_k) & set(relevant))

    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k=10):
    """Normalized Discounted Cumulative Gain."""

    top_k = recommended[:k]

    relevance = [
        1 if movie in relevant else 0
        for movie in top_k
    ]

    dcg = sum(
        rel / np.log2(i + 2)
        for i, rel in enumerate(relevance)
    )

    ideal_relevance = [1] * min(len(relevant), k)

    idcg = sum(
        rel / np.log2(i + 2)
        for i, rel in enumerate(ideal_relevance)
    )

    return dcg / idcg if idcg > 0 else 0.0


def recommendation_coverage(
    recommendations,
    total_catalog_movies
):
    """Fraction of movie catalog recommended at least once."""

    if total_catalog_movies <= 0:
        raise ValueError("Catalog size must be positive")

    unique_movies = recommendations["movieId"].nunique()

    return unique_movies / total_catalog_movies


def evaluate_user(
    recommended,
    relevant,
    k=10
):
    """Calculate ranking metrics for one user."""

    return {
        f"Precision@{k}": precision_at_k(
            recommended, relevant, k
        ),
        f"Recall@{k}": recall_at_k(
            recommended, relevant, k
        ),
        f"NDCG@{k}": ndcg_at_k(
            recommended, relevant, k
        )
    }
