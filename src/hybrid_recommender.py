
import pandas as pd


def calculate_alpha(num_interactions, k=68):
    """Calculate adaptive content and CF weights."""

    if num_interactions < 0:
        raise ValueError("Interactions cannot be negative.")

    alpha = k / (k + num_interactions)

    return alpha


def normalize_scores(score_dict):
    """Normalize recommendation scores between 0 and 1."""

    scores = pd.Series(score_dict, dtype=float)

    if scores.empty:
        return {}

    if scores.isna().any():
        raise ValueError("Scores cannot contain missing values.")

    min_score = scores.min()
    max_score = scores.max()

    if max_score == min_score:
        return {key: 1.0 for key in scores.index}

    normalized = (scores - min_score) / (max_score - min_score)

    return normalized.to_dict()


def hybrid_recommender(
    content_scores,
    cf_scores,
    num_interactions,
    top_k=10
):
    """Combine content and collaborative recommendation scores."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    alpha = calculate_alpha(num_interactions)

    content_norm = normalize_scores(content_scores)
    cf_norm = normalize_scores(cf_scores)

    all_movies = set(content_norm) | set(cf_norm)

    results = []

    for movie_id in all_movies:

        content = content_norm.get(movie_id, 0.0)
        cf = cf_norm.get(movie_id, 0.0)

        hybrid_score = (
            alpha * content +
            (1 - alpha) * cf
        )

        results.append({
            "movieId": movie_id,
            "content_score": content,
            "cf_score": cf,
            "hybrid_score": hybrid_score
        })

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        "hybrid_score",
        ascending=False
    )

    return result_df.head(top_k).reset_index(drop=True)
