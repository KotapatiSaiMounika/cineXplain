import pickle
import pandas as pd
from src.collaborative_recommender import get_cf_scores
from src.hybrid_recommender import hybrid_recommender

with open("data/model_outputs/temporal_knn/temporal_knn_model.pkl", "rb") as f:
    model = pickle.load(f)

content = pd.read_csv("data/model_outputs/content_scores_aligned.csv")
test = pd.read_csv("data/movielens/splits/test_ratings.csv")

users = sorted(
    set(test.loc[test.rating >= 4, "userId"]) & set(content.userId)
)

cf_hits = 0
hybrid_hits = 0

for i, uid in enumerate(users, 1):
    history = model.train[model.train.userId == uid]
    watched = set(history.movieId)

    relevant = set(test.loc[
        (test.userId == uid) & (test.rating >= 4), "movieId"
    ])

    cf = get_cf_scores(
        user_id=int(uid), method="knn", top_k=10000, model=model
    )

    cf_top10 = [
        mid for mid, score in sorted(
            cf.items(), key=lambda x: x[1], reverse=True
        )
        if mid not in watched
    ][:10]

    user_content = content[content.userId == uid]
    cs = dict(zip(user_content.movieId, user_content.content_score))

    hybrid = hybrid_recommender(
        cs, cf, len(history), top_k=10000, k=68
    )

    hybrid_top10 = hybrid[
        ~hybrid.movieId.isin(watched)
    ].head(10).movieId.tolist()

    cf_hits += len(set(cf_top10) & relevant)
    hybrid_hits += len(set(hybrid_top10) & relevant)

    if i % 200 == 0:
        print(f"Compared {i}/{len(users)} users")

n = len(users)

print("\nTEMPORAL MODEL COMPARISON")
print(f"Users evaluated: {n}")
print(f"CF-only Precision@10: {cf_hits / (n * 10):.5f}")
print(f"CF-only Recall@10: {cf_hits / n:.5f}")
print(f"Hybrid Precision@10: {hybrid_hits / (n * 10):.5f}")
print(f"Hybrid Recall@10: {hybrid_hits / n:.5f}")
