
import numpy as np
import pandas as pd


def calculate_alpha(num_interactions, k=68):
    """Calculate adaptive content and CF weights."""

    if num_interactions < 0:
        raise ValueError("Interactions cannot be negative.")

    if k <= 0:
        raise ValueError("k must be positive.")

    return k / (k + num_interactions)


def normalize_scores(score_dict):
    """Normalize dictionary scores between 0 and 1."""

    scores = pd.Series(score_dict, dtype=float)

    if scores.empty:
        return {}

    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite numbers.")

    minimum = scores.min()
    maximum = scores.max()

    if maximum == minimum:
        return {key: 1.0 for key in scores.index}

    return ((scores - minimum) / (maximum - minimum)).to_dict()


def normalize_column(series):
    """Normalize scores while preserving missing predictions."""

    result = pd.to_numeric(series, errors="raise").copy()
    valid = result.dropna()

    if not np.isfinite(valid).all():
        raise ValueError("Scores must be finite numbers.")

    if valid.empty:
        return result

    minimum = valid.min()
    maximum = valid.max()

    if minimum == maximum:
        result.loc[valid.index] = 1.0
    else:
        result.loc[valid.index] = (
            (valid - minimum) / (maximum - minimum)
        )

    return result


def hybrid_recommender(
    content_scores,
    cf_scores,
    num_interactions,
    top_k=10,
    k=68
):
    """Combine content and CF scores with adaptive weights."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    alpha = calculate_alpha(num_interactions, k)

    content_norm = normalize_scores(content_scores)
    cf_norm = normalize_scores(cf_scores)

    all_movies = set(content_norm) | set(cf_norm)

    results = []

    for movie_id in all_movies:

        content_available = movie_id in content_norm
        cf_available = movie_id in cf_norm

        content_weight = alpha if content_available else 0.0
        cf_weight = (1 - alpha) if cf_available else 0.0

        total_weight = content_weight + cf_weight

        # Reassign all weight to the available model
        content_weight /= total_weight
        cf_weight /= total_weight

        content = content_norm.get(movie_id, 0.0)
        cf = cf_norm.get(movie_id, 0.0)

        hybrid_score = (
            content_weight * content
            + cf_weight * cf
        )

        results.append({
            "movieId": movie_id,
            "content_score": content,
            "cf_score": cf,
            "content_weight": content_weight,
            "cf_weight": cf_weight,
            "hybrid_score": hybrid_score
        })

    return (
        pd.DataFrame(results)
        .sort_values("hybrid_score", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )


def generate_hybrid_recommendations(
    merged_scores,
    user_interactions,
    user_id,
    top_k=10,
    k=68
):
    """
    Generate personalized hybrid recommendations.

    Missing model predictions are handled by assigning
    the full available weight to the other model.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    required_scores = {
        "userId", "movieId", "content_score", "cf_score"
    }

    required_interactions = {
        "userId", "num_interactions"
    }

    if not required_scores.issubset(merged_scores.columns):
        raise ValueError("Missing required recommendation columns.")

    if not required_interactions.issubset(user_interactions.columns):
        raise ValueError("Missing required interaction columns.")

    user_data = merged_scores[
        merged_scores["userId"] == user_id
    ].copy()

    if user_data.empty:
        raise ValueError(
            f"No recommendations found for user {user_id}"
        )

    if user_data["movieId"].duplicated().any():
        raise ValueError("Duplicate movie predictions for this user.")

    interaction_row = user_interactions[
        user_interactions["userId"] == user_id
    ]

    if len(interaction_row) > 1:
        raise ValueError("Duplicate user interaction records.")

    num_interactions = (
        int(interaction_row["num_interactions"].iloc[0])
        if not interaction_row.empty else 0
    )

    alpha = calculate_alpha(num_interactions, k)

    # Normalize each model independently
    user_data["content_norm"] = normalize_column(
        user_data["content_score"]
    )

    user_data["cf_norm"] = normalize_column(
        user_data["cf_score"]
    )

    # Determine which models have predictions
    content_available = user_data["content_norm"].notna()
    cf_available = user_data["cf_norm"].notna()

    content_weight = alpha * content_available
    cf_weight = (1 - alpha) * cf_available

    total_weight = content_weight + cf_weight

    # Exclude movies with no prediction from either model
    valid = total_weight > 0

    user_data = user_data.loc[valid].copy()
    content_weight = content_weight.loc[valid]
    cf_weight = cf_weight.loc[valid]
    total_weight = total_weight.loc[valid]

    if user_data.empty:
        raise ValueError("No valid model predictions available.")

    # Re-normalize weights for available predictions
    user_data["content_weight"] = (
        content_weight / total_weight
    )

    user_data["cf_weight"] = (
        cf_weight / total_weight
    )

    user_data["hybrid_score"] = (
        user_data["content_norm"].fillna(0)
        * user_data["content_weight"]
        +
        user_data["cf_norm"].fillna(0)
        * user_data["cf_weight"]
    )

    user_data = user_data.sort_values(
        "hybrid_score",
        ascending=False
    )

    return user_data.head(top_k).reset_index(drop=True)
