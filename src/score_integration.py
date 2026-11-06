
import pandas as pd
from pathlib import Path


def load_content_scores(path):
    """Load and validate Member 1's predictions."""

    df = pd.read_csv(path)

    required = {"userId", "movieId", "content_score"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Content file must contain: {required}"
        )

    if df.empty:
        raise ValueError("Content score file is empty.")

    if df.duplicated(["userId", "movieId"]).any():
        raise ValueError("Duplicate user-movie pairs found.")

    if df["content_score"].isna().any():
        raise ValueError("Content scores contain missing values.")

    return df


def load_cf_scores(path):
    """Load and validate Member 2's predictions."""

    df = pd.read_csv(path)

    required = {"userId", "movieId", "cf_score"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"CF file must contain: {required}"
        )

    if df.empty:
        raise ValueError("CF score file is empty.")

    if df.duplicated(["userId", "movieId"]).any():
        raise ValueError("Duplicate user-movie pairs found.")

    if df["cf_score"].isna().any():
        raise ValueError("CF scores contain missing values.")

    return df


def integrate_scores(content_df, cf_df):

    merged = pd.merge(
        content_df,
        cf_df,
        on=["userId", "movieId"],
        how="outer"
    )

    print("Score integration completed.")
    print("Total candidate pairs:", len(merged))
    print("Content predictions:", merged["content_score"].notna().sum())
    print("CF predictions:", merged["cf_score"].notna().sum())

    return merged
