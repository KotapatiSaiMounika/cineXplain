
import pandas as pd


def prepare_for_mmr(hybrid_recommendations):
    """
    Prepare hybrid recommendations for Member 4's MMR module.
    """

    required_columns = {
        "movieId",
        "hybrid_score"
    }

    if not required_columns.issubset(
        hybrid_recommendations.columns
    ):
        raise ValueError("Required columns are missing.")

    result = hybrid_recommendations[
        ["movieId", "hybrid_score"]
    ].copy()

    result = result.dropna(
        subset=["movieId", "hybrid_score"]
    )

    result = result.sort_values(
        "hybrid_score",
        ascending=False
    )

    result = result.drop_duplicates(
        subset=["movieId"]
    )

    return result.reset_index(drop=True)
