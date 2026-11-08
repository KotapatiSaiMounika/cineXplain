
import pandas as pd

from evaluation.hybrid_metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k
)

# Load final recommendations
recommendations = pd.read_csv(
    "data/model_outputs/final_hybrid_recommendations.csv"
)

# Load testing ratings
test_data = pd.read_csv(
    "data/movielens/splits/test_ratings.csv"
)

user_id = 211

# Recommended movie IDs
recommended = recommendations["movieId"].tolist()

# Relevant movies (rating >= 4)
relevant = set(
    test_data.loc[
        (test_data["userId"] == user_id) &
        (test_data["rating"] >= 4),
        "movieId"
    ]
)

# Calculate metrics
precision = precision_at_k(recommended, relevant, k=10)
recall = recall_at_k(recommended, relevant, k=10)
ndcg = ndcg_at_k(recommended, relevant, k=10)

print("FINAL HYBRID EVALUATION")
print("Precision@10:", precision)
print("Recall@10:", recall)
print("NDCG@10:", ndcg)
