import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.content_recommender import load_movies, ContentRecommender

TRAIN = "data/movielens/splits/train_ratings.csv"
TEST = "data/movielens/splits/test_ratings.csv"
MOVIES = "data/movielens/movie.csv"
OUTPUT = "data/model_outputs/content_scores_aligned.csv"

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)

# Same eligible users as the hybrid evaluation
old_content = pd.read_csv("data/model_outputs/content_scores.csv")
eligible_users = sorted(
    set(test.loc[test.rating >= 4, "userId"])
    & set(old_content.userId)
)

movies = load_movies(
    movies_path=MOVIES,
    enriched_path="data/movielens/movies_enriched.csv"
)

rec = ContentRecommender(movies)
rec.fit_tfidf()

results = []

for i, user_id in enumerate(eligible_users, 1):
    history = train[train.userId == user_id]
    profile = dict(zip(history.movieId, history.rating))

    scores = rec.get_content_scores(
        profile,
        method="tfidf",
        top_k=100,
        exclude_seen=True
    )

    results.extend(
        {"userId": int(user_id), "movieId": mid, "content_score": score}
        for mid, score in scores.items()
    )

    if i % 100 == 0:
        print(f"Processed {i}/{len(eligible_users)} users")

pd.DataFrame(results).to_csv(OUTPUT, index=False)

print(f"Saved {len(results)} scores for {len(eligible_users)} users to {OUTPUT}")
