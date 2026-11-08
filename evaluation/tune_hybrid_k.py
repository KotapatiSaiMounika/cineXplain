import pandas as pd
from src.collaborative_recommender import get_cf_scores
from src.hybrid_recommender import hybrid_recommender

content = pd.read_csv("data/model_outputs/content_scores_aligned.csv")
test = pd.read_csv("data/movielens/splits/test_ratings.csv")
train = pd.read_csv(
    "data/movielens/splits/train_ratings.csv",
    usecols=["userId", "movieId", "rating"]
)

users = sorted(
    set(test.loc[test.rating >= 4, "userId"])
    & set(content.userId)
)

ks = [10, 25, 50, 68, 100, 150]
hits = {k: 0 for k in ks}
cf_hits = 0

for i, uid in enumerate(users, 1):
    history = train[train.userId == uid]
    watched = set(history.movieId)

    relevant = set(test.loc[
        (test.userId == uid) & (test.rating >= 4), "movieId"
    ])

    cf = get_cf_scores(int(uid), method="knn", top_k=10000)

    cf_top = [
        m for m, s in sorted(cf.items(), key=lambda x: x[1], reverse=True)
        if m not in watched
    ][:10]
    cf_hits += len(set(cf_top) & relevant)

    user_content = content[content.userId == uid]
    cs = dict(zip(user_content.movieId, user_content.content_score))

    for k in ks:
        recs = hybrid_recommender(
            cs, cf, len(history), top_k=10000, k=k
        )
        top10 = recs[~recs.movieId.isin(watched)].head(10)
        hits[k] += len(set(top10.movieId) & relevant)

    if i % 200 == 0:
        print(f"Evaluated {i}/{len(users)} users")

print("\nRESULTS")
print(f"CF-only Recall@10: {cf_hits / len(users):.5f}")
for k in ks:
    print(f"k={k:3}: Recall@10 = {hits[k] / len(users):.5f}")
