
import pandas as pd

from src.score_integration import integrate_scores


content = pd.DataFrame({
    "userId": [1, 1, 1],
    "movieId": [10, 20, 30],
    "content_score": [0.92, 0.85, 0.78]
})

cf = pd.DataFrame({
    "userId": [1, 1, 1],
    "movieId": [10, 20, 40],
    "cf_score": [4.5, 4.2, 3.8]
})

result = integrate_scores(content, cf)

print("\nIntegrated predictions:")
print(result)

from src.hybrid_recommender import (
    generate_hybrid_recommendations
)

# Load actual interaction counts
interactions = pd.read_csv(
    "data/movielens/user_interactions.csv"
)

# Generate recommendations for user 1
recommendations = generate_hybrid_recommendations(
    merged_scores=result,
    user_interactions=interactions,
    user_id=1,
    top_k=10
)

print("\nHYBRID RECOMMENDATIONS")
print(recommendations[
    [
        "userId",
        "movieId",
        "content_score",
        "cf_score",
        "hybrid_score",
        "content_weight",
        "cf_weight"
    ]
])
