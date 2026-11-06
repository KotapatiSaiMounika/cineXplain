
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

 
def normalize_column(series):
    """Normalize available scores to the 0-1 range."""

    result = series.copy()
    valid = series.dropna()

    if valid.empty:
        return result

    minimum = valid.min()
    maximum = valid.max()

    if maximum == minimum:
        result.loc[valid.index] = 1.0
    else:
        result.loc[valid.index] = (
            (valid - minimum) / (maximum - minimum)
        )

    return result


def generate_hybrid_recommendations(
    merged_scores,
    user_interactions,
    user_id,
    top_k=10,
    k=68
):
    """
    Generate personalized hybrid recommendations
    using content scores, CF scores and user history.
    """

    user_data = merged_scores[
        merged_scores["userId"] == user_id
    ].copy()

    if user_data.empty:
        raise ValueError(
            f"No recommendations found for user {user_id}"
        )

    # Get interaction count
    interaction_row = user_interactions[
        user_interactions["userId"] == user_id
    ]

    if interaction_row.empty:
        num_interactions = 0
    else:
        num_interactions = int(
            interaction_row["num_interactions"].iloc[0]
        )

    # Adaptive weights
    alpha = calculate_alpha(num_interactions, k)

    # Normalize separately, preserving missing predictions
    user_data["content_norm"] = normalize_column(
        user_data["content_score"]
    )

    user_data["cf_norm"] = normalize_column(
        user_data["cf_score"]
    )

    # Use only the models that actually have predictions
    content_available = user_data["content_norm"].notna()
    cf_available = user_data["cf_norm"].notna()

    content_weight = alpha * content_available
    cf_weight = (1 - alpha) * cf_available

    total_weight = content_weight + cf_weight

    user_data["hybrid_score"] = (
        user_data["content_norm"].fillna(0) * content_weight
        + user_data["cf_norm"].fillna(0) * cf_weight
    ) / total_weight

    user_data = user_data.sort_values(
        "hybrid_score",
        ascending=False
    )

    user_data["content_weight"] = (
        content_weight / total_weight
    )

    user_data["cf_weight"] = (
        cf_weight / total_weight
    )

    return user_data.head(top_k).reset_index(drop=True)

