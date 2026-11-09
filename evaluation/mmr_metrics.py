
"""
CineXplain - MMR Evaluation Metrics
Member 4

Metrics:
1. Intra-List Diversity (ILD)
2. Precision@K
3. NDCG@K
4. Comparison of Hybrid and MMR rankings
"""

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. INTRA-LIST DIVERSITY (ILD)
# --------------------------------------------------

def intra_list_diversity(
    recommended_ids,
    similarity_matrix,
    candidate_movie_ids
):
    """
    Measures diversity among recommended movies.

    ILD = 1 - Average Pairwise Similarity

    Higher ILD indicates greater diversity.

    Parameters
    ----------
    recommended_ids : list
        Movie IDs in the recommendation list.

    similarity_matrix : numpy array
        Pairwise cosine similarity matrix.

    candidate_movie_ids : list
        Movie IDs corresponding to the rows and columns
        of the similarity matrix.

    Returns
    -------
    float
        Intra-list diversity score.
    """

    if len(recommended_ids) < 2:
        return 0.0

    id_to_index = {
        movie_id: index
        for index, movie_id in enumerate(candidate_movie_ids)
    }

    indices = []

    for movie_id in recommended_ids:
        if movie_id not in id_to_index:
            raise ValueError(
                f"Movie ID {movie_id} not found in candidate_movie_ids"
            )
        indices.append(id_to_index[movie_id])

    similarities = []

    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            similarities.append(
                similarity_matrix[indices[i], indices[j]]
            )

    if not similarities:
        return 0.0

    return float(1 - np.mean(similarities))


# --------------------------------------------------
# 2. PRECISION@K
# --------------------------------------------------

def precision_at_k(recommended_ids, relevant_movie_ids, k=10):
    """
    Calculates Precision@K.

    Precision@K =
    Relevant recommended movies / K

    Higher precision indicates better relevance.
    """

    if k <= 0:
        raise ValueError("K must be greater than zero.")

    if not relevant_movie_ids:
        return 0.0

    top_k = recommended_ids[:k]

    relevant_count = sum(
        movie_id in relevant_movie_ids
        for movie_id in top_k
    )

    return relevant_count / k


# --------------------------------------------------
# 3. NDCG@K
# --------------------------------------------------

def ndcg_at_k(recommended_ids, relevance_scores, k=10):
    """
    Calculates Normalized Discounted Cumulative Gain.

    relevance_scores:
        Dictionary containing movie IDs and their
        ground-truth relevance grades.

    Higher NDCG indicates better ranking quality.
    """

    if k <= 0:
        raise ValueError("K must be greater than zero.")

    if any(score < 0 for score in relevance_scores.values()):
        raise ValueError("Relevance scores cannot be negative.")

    top_k = recommended_ids[:k]

    # Relevance scores in recommendation order
    actual_relevance = [
        relevance_scores.get(movie_id, 0)
        for movie_id in top_k
    ]

    def calculate_dcg(scores):
        return sum(
            (2 ** score - 1) / np.log2(position + 2)
            for position, score in enumerate(scores)
        )

    dcg = calculate_dcg(actual_relevance)

    # Ideal ranking: highest relevance first
    ideal_relevance = sorted(
        relevance_scores.values(),
        reverse=True
    )[:k]

    idcg = calculate_dcg(ideal_relevance)

    if idcg == 0:
        return 0.0

    return float(dcg / idcg)


# --------------------------------------------------
# 4. EVALUATE A RECOMMENDATION LIST
# --------------------------------------------------

def evaluate_recommendations(
    recommended_ids,
    relevant_movie_ids,
    relevance_scores,
    similarity_matrix,
    candidate_movie_ids,
    k=10
):
    """
    Calculates all MMR evaluation metrics
    for a recommendation list.
    """

    return {
        "ILD": intra_list_diversity(
            recommended_ids[:k],
            similarity_matrix,
            candidate_movie_ids
        ),

        "Precision@K": precision_at_k(
            recommended_ids,
            relevant_movie_ids,
            k
        ),

        "NDCG@K": ndcg_at_k(
            recommended_ids,
            relevance_scores,
            k
        )
    }


# --------------------------------------------------
# 5. COMPARE HYBRID VS MMR
# --------------------------------------------------

def compare_recommendations(
    hybrid_ids,
    mmr_ids,
    relevant_movie_ids,
    relevance_scores,
    similarity_matrix,
    candidate_movie_ids,
    k=10
):
    """
    Compares the original hybrid ranking with
    the MMR re-ranked recommendation list.
    """

    hybrid_metrics = evaluate_recommendations(
        hybrid_ids,
        relevant_movie_ids,
        relevance_scores,
        similarity_matrix,
        candidate_movie_ids,
        k
    )

    mmr_metrics = evaluate_recommendations(
        mmr_ids,
        relevant_movie_ids,
        relevance_scores,
        similarity_matrix,
        candidate_movie_ids,
        k
    )

    comparison = pd.DataFrame(
        {
            "Hybrid Ranking": hybrid_metrics,
            "MMR Ranking": mmr_metrics
        }
    )

    comparison["Improvement"] = (
        comparison["MMR Ranking"]
        - comparison["Hybrid Ranking"]
    )

    return comparison


# --------------------------------------------------
# 6. LAMBDA EXPERIMENT RESULTS
# --------------------------------------------------

def compare_lambda_results(results):
    """
    Creates a comparison table for different lambda values.

    Expected format:

    results = {
        0.3: {"ILD": 0.8, "Precision@K": 0.6, "NDCG@K": 0.7},
        0.5: {"ILD": 0.7, "Precision@K": 0.7, "NDCG@K": 0.8}
    }
    """

    df = pd.DataFrame.from_dict(
        results,
        orient="index"
    )

    df.index.name = "Lambda"

    return df.sort_index()


# --------------------------------------------------
# 7. MAIN DEMONSTRATION
# --------------------------------------------------

if __name__ == "__main__":

    # Demonstration data only.
    # Replace with actual outputs from CineXplain.

    candidate_movie_ids = [101, 102, 103, 104, 105]

    similarity_matrix = np.array([
        [1.0, 0.9, 0.2, 0.1, 0.3],
        [0.9, 1.0, 0.3, 0.2, 0.4],
        [0.2, 0.3, 1.0, 0.8, 0.2],
        [0.1, 0.2, 0.8, 1.0, 0.3],
        [0.3, 0.4, 0.2, 0.3, 1.0]
    ])

    hybrid_ids = [101, 102, 103, 104, 105]
    mmr_ids = [101, 103, 105, 102, 104]

    relevant_movie_ids = {101, 102, 103}

    relevance_scores = {
        101: 3,
        102: 3,
        103: 2,
        104: 1,
        105: 0
    }

    comparison = compare_recommendations(
        hybrid_ids,
        mmr_ids,
        relevant_movie_ids,
        relevance_scores,
        similarity_matrix,
        candidate_movie_ids,
        k=5
    )

    print("\nCineXplain: Hybrid vs MMR Evaluation")
    print("-------------------------------------")
    print(comparison.round(4))

    print("\nLambda Experiment Demonstration")

    lambda_results = {
        0.3: {"ILD": 0.80, "Precision@K": 0.60, "NDCG@K": 0.70},
        0.5: {"ILD": 0.70, "Precision@K": 0.70, "NDCG@K": 0.80},
        0.7: {"ILD": 0.60, "Precision@K": 0.80, "NDCG@K": 0.85},
        0.9: {"ILD": 0.40, "Precision@K": 0.90, "NDCG@K": 0.92}
    }

    print(compare_lambda_results(lambda_results).round(4))
