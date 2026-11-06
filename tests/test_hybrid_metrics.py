
from evaluation.hybrid_metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    evaluate_user
)

recommended = [101, 102, 103, 104, 105]

relevant = {101, 103, 105, 108}

print("PRECISION:", precision_at_k(
    recommended, relevant, k=5
))

print("RECALL:", recall_at_k(
    recommended, relevant, k=5
))

print("NDCG:", ndcg_at_k(
    recommended, relevant, k=5
))

print("\nComplete evaluation:")

print(evaluate_user(
    recommended,
    relevant,
    k=5
))
