
from src.collaborative_recommender import get_cf_scores
from src.hybrid_recommender import hybrid_recommender

# 1. Get actual collaborative predictions
cf_scores = get_cf_scores(
    user_id=1,
    method="knn",
    top_k=100
)

print("CF predictions:", len(cf_scores))

# 2. Temporary content scores
# We will replace these with Member 1's real predictions.
content_scores = {
    movie_id: 0.8
    for movie_id in cf_scores.keys()
}

# 3. Generate hybrid recommendations
recommendations = hybrid_recommender(
    content_scores=content_scores,
    cf_scores=cf_scores,
    num_interactions=175,
    top_k=10
)

print("\nCINEXPLAIN HYBRID RECOMMENDATIONS")
print(recommendations)

# 4. Verify
assert len(recommendations) > 0
assert recommendations["hybrid_score"].notna().all()

print("\nREAL CF + HYBRID INTEGRATION PASSED!")
