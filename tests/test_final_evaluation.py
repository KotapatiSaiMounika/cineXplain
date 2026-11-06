
import pandas as pd

from evaluation.hybrid_metrics import evaluate_all_users

# Sample recommendations
recommendations = pd.DataFrame({
    "userId": [1, 1, 1, 2, 2, 2],
    "movieId": [10, 20, 30, 40, 50, 60],
    "hybrid_score": [0.95, 0.85, 0.70, 0.90, 0.80, 0.60]
})

# Sample held-out ratings
test_ratings = pd.DataFrame({
    "userId": [1, 2],
    "movieId": [10, 50],
    "rating": [5.0, 4.5]
})

results, averages = evaluate_all_users(
    recommendations,
    test_ratings,
    k=3
)

print("INDIVIDUAL USER RESULTS")
print(results)

print("\nAVERAGE PERFORMANCE")
print(averages)
