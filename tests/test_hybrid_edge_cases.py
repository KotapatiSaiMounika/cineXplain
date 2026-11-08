
import pandas as pd

from src.hybrid_recommender import (
    calculate_alpha,
    generate_hybrid_recommendations
)

# Test 1: Cold-start user
assert calculate_alpha(0) == 1.0

# Test 2: Experienced user
assert calculate_alpha(1000) < 0.1

# Test 3: Missing CF prediction
data = pd.DataFrame({
    "userId": [1, 1, 1],
    "movieId": [101, 102, 103],
    "content_score": [0.9, 0.8, 0.7],
    "cf_score": [None, 4.5, 3.5]
})

interactions = pd.DataFrame({
    "userId": [1],
    "num_interactions": [20]
})

result = generate_hybrid_recommendations(
    data,
    interactions,
    user_id=1,
    top_k=3
)

assert not result["hybrid_score"].isna().any()
assert result.loc[
    result["movieId"] == 101,
    "content_weight"
].iloc[0] == 1.0

# Test 4: Ranking order
assert result["hybrid_score"].is_monotonic_decreasing

print("ALL HYBRID EDGE CASE TESTS PASSED!")
print(result[[
    "movieId",
    "hybrid_score",
    "content_weight",
    "cf_weight"
]])
