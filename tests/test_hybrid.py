from src.hybrid_recommender import (
    calculate_alpha,
    hybrid_recommender
)

content_scores = {
    101: 0.90,
    102: 0.80,
    103: 0.95,
    104: 0.70
}

cf_scores = {
    101: 4.5,
    102: 4.8,
    103: 3.5,
    104: 4.9
}

print("Content weight:", calculate_alpha(20))

recommendations = hybrid_recommender(
    content_scores,
    cf_scores,
    num_interactions=20,
    top_k=4
)

print(recommendations)