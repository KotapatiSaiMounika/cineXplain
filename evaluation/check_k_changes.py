import pandas as pd
from src.collaborative_recommender import get_cf_scores
from src.hybrid_recommender import hybrid_recommender

content = pd.read_csv("data/model_outputs/content_scores_aligned.csv")
test = pd.read_csv("data/movielens/splits/test_ratings.csv")
train = pd.read_csv(
    "data/movielens/splits/train_ratings.csv",
    usecols=["userId", "movieId"]
)

users = sorted(
    set(test.loc[test.rating >= 4, "userId"]) & set(content.userId)
)
ks = [10, 25, 50, 68, 100, 150]
changed = {k: 0 for k in ks if k != 68}

for i, uid in enumerate(users, 1):
    history = train[train.userId == uid]
    watched = set(history.movieId)
    cf = get_cf_scores(int(uid), method="knn", top_k=10000)

    user_content = content[content.userId == uid]
    cs = dict(zip(user_content.movieId, user_content.content_score))

    lists = {}
    for k in ks:
        recs = hybrid_recommender(
            cs, cf, len(history), top_k=10000, k=k
        )
        lists[k] = tuple(
            recs[~recs.movieId.isin(watched)].head(10).movieId
        )

    for k in changed:
        if lists[k] != lists[68]:
            changed[k] += 1

    if i % 200 == 0:
        print(f"Checked {i}/{len(users)}")

print("\nUsers with a different top-10 from k=68:")
for k, count in changed.items():
    print(f"k={k}: {count}/{len(users)}")
