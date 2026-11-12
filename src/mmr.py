
"""
CineXplain - MMR Re-ranking Module
Member 4: Recommendation Diversity

Dataset: movies_enriched.csv

Pipeline:
1. Load enriched movie metadata.
2. Build movie text representations.
3. Generate TF-IDF vectors.
4. Calculate pairwise cosine similarity.
5. Apply greedy Maximal Marginal Relevance.
6. Return diversified recommendations.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. DATASET CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATHS = [
    PROJECT_ROOT / "data" / "movies_enriched.csv",
]

# These are automatically selected if present.
DEFAULT_FEATURE_COLUMNS = [
    "title",
    "overview",
    "plot",
    "genres",
    "cast",
    "keywords",
    "director",
    "tagline",
    "release_year",
    "year",
]


# --------------------------------------------------
# 2. LOAD ENRICHED MOVIE DATASET
# --------------------------------------------------

def load_movies(dataset_path=None):
    """
    Load the enriched movie dataset.

    Parameters
    ----------
    dataset_path : str or Path, optional
        Path to movies_enriched.csv.

    Returns
    -------
    pandas.DataFrame
    """

    if dataset_path is None:
        dataset_path = next(
            (path for path in DATASET_PATHS if path.exists()),
            None
        )

    if dataset_path is None:
        raise FileNotFoundError(
            "Movie dataset not found. Expected movies_enriched.csv "
            "inside the project's data/ folder."
        )

    movies = pd.read_csv(dataset_path)

    movies.columns = movies.columns.str.strip()

    if movies.empty:
        raise ValueError("The movie dataset is empty.")

    print("Dataset loaded successfully!")
    print("Shape:", movies.shape)
    print("Columns:", movies.columns.tolist())

    return movies


# --------------------------------------------------
# 3. PREPARE MOVIE TEXT
# --------------------------------------------------

def prepare_movie_text(
    movies,
    feature_columns=None
):
    """
    Combine movie metadata into a single text document.

    Features can include:
    title, plot, genres, cast, director and year.

    Returns
    -------
    pandas.Series
        Combined text representation for each movie.
    """

    if feature_columns is None:
        feature_columns = [
            col for col in DEFAULT_FEATURE_COLUMNS
            if col in movies.columns
        ]

    if not feature_columns:
        raise ValueError(
            "No supported text feature columns found. "
            "Pass feature_columns explicitly."
        )

    missing = [
        col for col in feature_columns
        if col not in movies.columns
    ]

    if missing:
        raise ValueError(
            f"Missing feature columns: {missing}"
        )

    print("Features used:", feature_columns)

    text = movies[feature_columns].fillna("").astype(str)

    combined_text = text.agg(" ".join, axis=1)

    return combined_text


# --------------------------------------------------
# 4. BUILD TF-IDF SIMILARITY MATRIX
# --------------------------------------------------

def build_similarity_matrix(
    movies,
    candidate_movie_ids,
    id_col="movieId",
    feature_columns=None
):
    """
    Calculate pairwise cosine similarity between candidate movies.

    Only candidate movies are vectorized to avoid constructing
    a large all-movie similarity matrix.

    Returns
    -------
    numpy.ndarray
        Similarity matrix of shape (n, n).
    """

    if id_col not in movies.columns:
        raise ValueError(
            f"Movie ID column '{id_col}' not found."
        )

    candidate_movie_ids = list(candidate_movie_ids)

    if len(set(candidate_movie_ids)) != len(candidate_movie_ids):
        raise ValueError("Candidate movie IDs must be unique.")

    indexed_movies = movies.set_index(id_col, drop=False)

    missing_ids = [
        movie_id for movie_id in candidate_movie_ids
        if movie_id not in indexed_movies.index
    ]

    if missing_ids:
        raise ValueError(
            f"Movie IDs missing from dataset: {missing_ids[:10]}"
        )

    candidate_movies = indexed_movies.loc[
        candidate_movie_ids
    ].copy()

    if candidate_movies.index.duplicated().any():
        raise ValueError("Movie IDs must be unique in the dataset.")

    combined_text = prepare_movie_text(
        candidate_movies,
        feature_columns
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000
    )

    tfidf_matrix = vectorizer.fit_transform(combined_text)

    similarity_matrix = cosine_similarity(tfidf_matrix)

    print("TF-IDF matrix shape:", tfidf_matrix.shape)
    print("Similarity matrix shape:", similarity_matrix.shape)

    return similarity_matrix


# --------------------------------------------------
# 5. MAXIMAL MARGINAL RELEVANCE
# --------------------------------------------------

def mmr_rerank(
    movie_ids,
    relevance_scores,
    similarity_matrix,
    k=10,
    lambda_param=0.7
):
    """
    Greedy Maximal Marginal Relevance algorithm.

    MMR(d) = lambda * Relevance(d)
             - (1-lambda) * MaxSimilarity(d, selected)

    Returns
    -------
    list
        (movie_id, relevance_score, mmr_score)
    """

    movie_ids = list(movie_ids)

    relevance_scores = np.asarray(
        relevance_scores,
        dtype=float
    )

    similarity_matrix = np.asarray(
        similarity_matrix,
        dtype=float
    )

    n = len(movie_ids)

    # Input validation
    if n == 0:
        return []

    if len(set(movie_ids)) != n:
        raise ValueError("Duplicate movie IDs found.")

    if relevance_scores.shape != (n,):
        raise ValueError("Invalid relevance score dimensions.")

    if similarity_matrix.shape != (n, n):
        raise ValueError("Invalid similarity matrix dimensions.")

    if not np.all(np.isfinite(relevance_scores)):
        raise ValueError("Relevance scores must be finite.")

    if not np.all(np.isfinite(similarity_matrix)):
        raise ValueError("Similarity matrix must be finite.")

    if not 0 <= lambda_param <= 1:
        raise ValueError("Lambda must be between 0 and 1.")

    if not np.all((0 <= relevance_scores) & (relevance_scores <= 1)):
        raise ValueError("Relevance scores must be in [0, 1].")

    if not np.all((0 <= similarity_matrix) & (similarity_matrix <= 1)):
        raise ValueError("Similarity scores must be in [0, 1].")

    if not np.allclose(similarity_matrix, similarity_matrix.T):
        raise ValueError("Similarity matrix must be symmetric.")

    if not np.allclose(np.diag(similarity_matrix), 1.0):
        raise ValueError("Similarity matrix diagonal must be 1.")

    if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
        raise TypeError("k must be an integer.")

    if k <= 0:
        raise ValueError("k must be positive.")

    k = min(k, n)

    # Initialization
    selected = []
    remaining = list(range(n))
    selected_scores = []

    # Greedy MMR selection
    while len(selected) < k:

        best_idx = None
        best_score = -np.inf

        for idx in remaining:

            relevance = relevance_scores[idx]

            if not selected:
                redundancy = 0.0

            else:
                redundancy = np.max(
                    similarity_matrix[idx, selected]
                )

            mmr_score = (
                lambda_param * relevance
                - (1 - lambda_param) * redundancy
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        selected_scores.append(float(best_score))

        remaining.remove(best_idx)

    return [
        (
            movie_ids[idx],
            float(relevance_scores[idx]),
            selected_scores[position]
        )
        for position, idx in enumerate(selected)
    ]


# --------------------------------------------------
# 6. COMPLETE MMR PIPELINE
# --------------------------------------------------

def recommend_diverse_movies(
    movies,
    movie_ids,
    relevance_scores,
    k=10,
    lambda_param=0.7,
    id_col="movieId",
    feature_columns=None
):
    """
    Complete MMR pipeline using enriched movie metadata
    and hybrid recommendation scores.

    Returns
    -------
    pandas.DataFrame
    """

    similarity_matrix = build_similarity_matrix(
        movies=movies,
        candidate_movie_ids=movie_ids,
        id_col=id_col,
        feature_columns=feature_columns
    )

    recommendations = mmr_rerank(
        movie_ids=movie_ids,
        relevance_scores=relevance_scores,
        similarity_matrix=similarity_matrix,
        k=k,
        lambda_param=lambda_param
    )

    result = pd.DataFrame(
        recommendations,
        columns=[
            id_col,
            "relevance_score",
            "mmr_score"
        ]
    )

    # Add movie titles and other metadata.
    result = result.merge(
        movies,
        on=id_col,
        how="left"
    )

    return result


# --------------------------------------------------
# 7. EXECUTION EXAMPLE
# --------------------------------------------------

if __name__ == "__main__":

    movies = load_movies()

    print("\nDataset preview:")
    print(movies.head())

    print(
        "\nDataset loaded. Waiting for hybrid recommendation "
        "scores from Member 3."
    )

    # Temporary demonstration only.
    # Replace these IDs and scores with Member 3's output.

    id_col = "movieId"

    if id_col in movies.columns:

        sample = movies.dropna(subset=[id_col]).head(20)

        candidate_ids = sample[id_col].tolist()

        sample_scores = np.linspace(
            0.95, 0.50, len(candidate_ids)
        )

        recommendations = recommend_diverse_movies(
            movies=movies,
            movie_ids=candidate_ids,
            relevance_scores=sample_scores,
            k=min(10, len(candidate_ids)),
            lambda_param=0.7,
            id_col=id_col
        )

        print("\nCineXplain MMR Recommendations:")
        print(
            recommendations[
                [
                    id_col,
                    "title",
                    "relevance_score",
                    "mmr_score"
                ]
            ].to_string(index=False)
        )
