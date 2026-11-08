import pandas as pd
from src.collaborative_recommender import get_cf_scores
from src.hybrid_recommender import hybrid_recommender

content = pd.read_csv("data/model_outputs/content_scores_aligned.csv")
test = pd.read_csv("data/movielens/splits/test_ratings.csv")
train = pd.read_csv(
    "data/movielens/splits/train_ratings.csv",
    usecols=["userId", "movieId", "rating"]
)

users = sorted(set(test.loc[test.rating >= 4, "userId"]) & set(content.userId))
hits_cf = 0
hits_hybrid = 0

for i, uid in enumerate(users, 1):
    relevant = set(test.loc[
        (test.userId == uid) & (test.rating >= 4), "movieId"
    ])
    history = train[train.userId == uid]
    watched = set(history.movieId)

    cf = get_cf_scores(int(uid), method="knn", top_k=10000)
    user_content = content[content.userId == uid]
    cs = dict(zip(user_content.movieId, user_content.content_score))

    cf_top = [
        m for m, s in sorted(cf.items(), key=lambda x: x[1], reverse=True)
        if m not in watched
    ][:10]

    hybrid = hybrid_recommender(
        cs, cf, len(history), top_k=10000
    )
    hybrid_top = hybrid[~hybrid.movieId.isin(watched)].head(10).movieId.tolist()

    hits_cf += len(set(cf_top) & relevant)
    hits_hybrid += len(set(hybrid_top) & relevant)

    if i % 200 == 0:
        print(f"Compared {i}/{len(users)} users")

print("\nCF-only Precision@10:", hits_cf / (len(users) * 10))
print("CF-only Recall@10:", hits_cf / len(users))
print("Hybrid Precision@10:", hits_hybrid / (len(users) * 10))
print("Hybrid Recall@10:", hits_hybrid / len(users))
