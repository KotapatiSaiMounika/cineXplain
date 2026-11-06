
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/movielens")
OUTPUT_DIR = DATA_DIR / "splits"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load actual MovieLens ratings
ratings = pd.read_csv(DATA_DIR / "rating.csv")

# Convert timestamp to datetime
ratings["timestamp"] = pd.to_datetime(
    ratings["timestamp"]
)

# Sort each user's ratings chronologically
ratings = ratings.sort_values(
    ["userId", "timestamp"]
)

# Hold out each user's latest rating
test_ratings = ratings.groupby(
    "userId"
).tail(1)

# Earlier ratings become training data
train_ratings = ratings.drop(
    test_ratings.index
)

# Save datasets
train_ratings.to_csv(
    OUTPUT_DIR / "train_ratings.csv",
    index=False
)

test_ratings.to_csv(
    OUTPUT_DIR / "test_ratings.csv",
    index=False
)

print("TRAIN/TEST SPLIT COMPLETED")
print("=" * 40)

print("Total ratings:", len(ratings))
print("Training ratings:", len(train_ratings))
print("Testing ratings:", len(test_ratings))

print(
    "Unique training users:",
    train_ratings["userId"].nunique()
)

print(
    "Unique testing users:",
    test_ratings["userId"].nunique()
)

print("\nFiles saved successfully!")
