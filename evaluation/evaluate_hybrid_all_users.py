
import pandas as pd

from src.collaborative_recommender import get_cf_scores
from src.hybrid_recommender import hybrid_recommender

content = pd.read_csv(
    "data/model_outputs/content_scores_aligned.csv"
   
)

test = pd.read_csv(
    "data/movielens/splits/test_ratings.csv"
)

train = pd.read_csv(
    "data/movielens/splits/train_ratings.csv",
    usecols=["userId", "movieId"]
)

# Users with relevant held-out ratings
relevant_users = set(
    test.loc[test.rating >= 4, "userId"]
)

content_users = set(content.userId)

eligible_users = sorted(
    relevant_users & content_users
)

results = []

for i, user_id in enumerate(eligible_users, 1):

    user_content = content[
        content.userId == user_id
    ]

    content_scores = dict(zip(
        user_content.movieId,
        user_content.content_score
    ))

    cf_scores = get_cf_scores(
        user_id=int(user_id),
        method="knn",
        top_k=10000
    )

    interactions = train[
        train.userId == user_id
    ]

    num_interactions = len(interactions)

    recommendations = hybrid_recommender(
        content_scores,
        cf_scores,
        num_interactions,
        top_k=10000
    )

    watched = set(interactions.movieId)

    recommendations = recommendations[
        ~recommendations.movieId.isin(watched)
    ].head(10)

    relevant = set(
        test.loc[
            (test.userId == user_id) &
            (test.rating >= 4),
            "movieId"
        ]
    )

    recommended = recommendations.movieId.tolist()

    hits = len(set(recommended) & relevant)

    precision = hits / 10
    recall = hits / len(relevant)

    results.append({
        "userId": user_id,
        "precision": precision,
        "recall": recall,
        "hits": hits
    })

    if i % 100 == 0:
        print(f"Evaluated {i}/{len(eligible_users)} users")

df = pd.DataFrame(results)

print("\nFINAL HYBRID EVALUATION")
print("Evaluated users:", len(df))
print("Precision@10:", df.precision.mean())
print("Recall@10:", df.recall.mean())

df.to_csv(
    "reports/hybrid_all_users_results.csv",
    index=False
)

print("\nResults saved successfully!")
