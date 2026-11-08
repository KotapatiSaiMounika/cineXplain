
import pandas as pd

from src.hybrid_interface import prepare_for_mmr

sample = pd.DataFrame({
    "movieId": [101, 102, 103, 104],
    "hybrid_score": [0.95, 0.85, 0.75, 0.65],
    "content_score": [0.9, 0.8, 0.7, 0.6]
})

result = prepare_for_mmr(sample)

print("MMR INPUT:")
print(result)

assert list(result.columns) == [
    "movieId", "hybrid_score"
]

assert result["hybrid_score"].is_monotonic_decreasing

print("\nMMR interface test passed!")
