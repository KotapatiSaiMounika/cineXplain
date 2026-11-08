
import pandas as pd

from src.collaborative_recommender import get_cf_scores
from src.hybrid_recommender import hybrid_recommender

# Select user
user_id = 211

# Load actual content predictions
content_df = pd.read_csv(
    "data/model_outputs/content_scores.csv"
)

user_content = content_df[
    content_df["userId"] == user_id
]

content_scores = dict(zip(
    user_content["movieId"],
    user_content["content_score"]
))

# Get actual collaborative predictions
cf_scores = get_cf_scores(
    user_id=user_id,
    method="knn",
    top_k=10000
)

# Get user's interaction count
interactions = pd.read_csv(
    "data/movielens/user_interactions.csv"
)

user_row = interactions[
    interactions["userId"] == user_id
]

num_interactions = (
    int(user_row["num_interactions"].iloc[0])
    if not user_row.empty else 0
)

# Generate hybrid rankings
recommendations = hybrid_recommender(
    content_scores,
    cf_scores,
    num_interactions=num_interactions,
    top_k=200
)


# Load training history only
train_ratings = pd.read_csv(
    "data/movielens/splits/train_ratings.csv",
    usecols=["userId", "movieId"]
)

# Calculate user interaction count
num_interactions = len(
    train_ratings[
        train_ratings["userId"] == user_id
    ]
)

# Get movies already seen during training
watched_movies = set(
    train_ratings.loc[
        train_ratings["userId"] == user_id,
        "movieId"
    ]
)

# Remove watched movies
recommendations = recommendations[
    ~recommendations["movieId"].isin(watched_movies)
].head(10)

# Save recommendations
recommendations.to_csv(
    "data/model_outputs/final_hybrid_recommendations.csv",
    index=False
)

print("\nFINAL CINEXPLAIN RECOMMENDATIONS")
print(recommendations)

print("\nSaved successfully!")
