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

results = []

for i, uid in enumerate(users, 1):
    history = model.train[model.train.userId == uid]
    watched = set(history.movieId)

    relevant = set(test.loc[
        (test.userId == uid) & (test.rating >= 4), "movieId"
    ])

    if uid not in model.user_to_idx:
        continue

    cf = get_cf_scores(
        user_id=int(uid),
        method="knn",
        top_k=10000,
        model=model
    )

    user_content = content[content.userId == uid]
    cs = dict(zip(user_content.movieId, user_content.content_score))

    recs = hybrid_recommender(
        cs, cf, len(history), top_k=10000, k=68
    )

    top10 = recs[~recs.movieId.isin(watched)].head(10)
    hits = len(set(top10.movieId) & relevant)

    results.append({
        "userId": uid,
        "precision": hits / 10,
        "recall": hits / len(relevant),
        "hits": hits
    })

    if i % 200 == 0:
        print(f"Evaluated {i}/{len(users)} users")

df = pd.DataFrame(results)
print("\nTEMPORAL HYBRID RESULTS")
print("Users:", len(df))
print("Precision@10:", df.precision.mean())
print("Recall@10:", df.recall.mean())

df.to_csv("reports/temporal_hybrid_results.csv", index=False)
print("Saved reports/temporal_hybrid_results.csv")
