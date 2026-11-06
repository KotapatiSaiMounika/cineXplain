# Member 2: Collaborative Filtering

## What it does

Recommends movies from other users' rating behaviour, using two algorithms built only from TRAIN ratings (no leakage):

- **Item-Item KNN**: adjusted-cosine similarity between movies (each user's mean rating removed), top-50 neighbours per movie. A movie's score is its summed similarity to the movies the user rated 4 or higher.
- **SVD**: biased matrix factorization, `rating ~ mu + b_user + b_movie + p_user . q_movie`, 50 latent factors, trained with full-batch gradient descent (Adam) and L2 regularisation. Ranking score = predicted rating + 0.5 * log(popularity).

**Working set:** MovieLens ML-32M restricted to movies with at least 50 ratings (16,034 movies) and a random sample of 30,000 users. Split: per-user random 80/20, seed 42. "Relevant" = test rating >= 4.0.

## How to run

```bash
pip install -r requirements.txt
python evaluation/collaborative_metrics.py     # metric self-test
python src/collaborative_recommender.py        # demo, interface checks, evaluation
```

The first run builds the working set and models (about 1-2 minutes) and caches them in `data/cache/` (gitignored). Later runs take seconds. The full experiments are in `notebooks/02_collaborative_filtering.ipynb`.

## Interface for the hybrid model (Member 3)

```python
from src.collaborative_recommender import get_cf_scores

scores = get_cf_scores(user_id=5, method="knn", top_k=100)
# -> {movieId: score in [0, 1], ...} best first

# user that is not in the data: pass a profile in the same {movieId: rating} form as Member 1
scores = get_cf_scores(rating_history={318: 5.0, 296: 4.5, 50: 2.0}, method="svd")
```

- `method` is `"knn"` or `"svd"`; `normalize=False` returns raw scores.
- Already-rated movies are never returned.
- Unknown users, or fewer than 5 usable ratings, get popularity-based scores (cold start). Unknown movieIds in a profile are ignored.
- **Candidates are only the 16,034 movies with >= 50 ratings**, not the full 87,585-movie catalog. The hybrid must handle movies that have a content score but no collaborative score.
- Other helpers: `recommend_knn()`, `recommend_svd()`, `predict_svd(user_id, movie_id)` (star rating), `get_model()`.

## Evaluation

Same test data for every model; ranking metrics use 29,611 users with at least one relevant test movie; candidates are all movies the user has not rated in train. Full table: `evaluation/results_collaborative.csv`.

| | Item-KNN (K=50) | SVD (50 factors) | Popularity |
|---|---|---|---|
| RMSE | 0.913 | 0.796 | - |
| MAE | 0.701 | 0.601 | - |
| Precision@10 | 0.215 | 0.144 | 0.119 |
| Recall@10 | 0.184 | 0.113 | 0.099 |
| NDCG@10 | 0.288 | 0.184 | 0.154 |
| Training time | 13 s | 38 s | - |

Experiments:

- **KNN neighbours:** K=50 ranks best (NDCG@10 0.288); K=100 has slightly lower RMSE (0.906) but worse ranking.
- **SVD factors:** 10 / 25 / 50 / 100 factors give RMSE 0.817 / 0.805 / 0.796 / 0.791 and NDCG@10 0.167 / 0.177 / 0.184 / 0.190. Returns diminish while training time grows (15 s to 64 s), so 50 is the default.
- **By user history (NDCG@10):** light users (<30 ratings) KNN 0.186, SVD 0.103; medium (30-100) 0.236 vs 0.136; heavy (>100) 0.432 vs 0.309. KNN is best in every group.

SVD predicts star ratings better; KNN ranks better.

## Limitations

- **Working set, not the full catalog:** 30,000 users and 16,034 movies, so recommendations cover about 18% of the catalog.
- **KNN RMSE is not like for like.** KNN can only predict a rating with neighbour evidence (88.5% of test ratings at K=50); the rest fall back to the user's mean. SVD covers 100%.
- **Plain SVD ranking is poor.** Ranking by predicted rating alone gave Precision@10 of 0.003, because rarely-rated movies get noisy scores (38% of movies have under 20 training ratings in this sample). The popularity term fixes this, but SVD lists lean toward blockbusters.
- **Cold-start users** get the same popularity list.
- **Single split and seed,** as in the content module. Repeated splits would give error bars.
- **Split protocol:** same strategy as the content module (per-user random 80/20, seed 42, relevant >= 4), but a different user sample (30,000 users vs. 200), so the numbers are not directly comparable across modules.

## Files

| File | Purpose |
|---|---|
| `src/collaborative_recommender.py` | Data split, KNN similarity, SVD training, `get_cf_scores()` and helpers |
| `evaluation/collaborative_metrics.py` | RMSE, MAE, Precision@K, Recall@K, NDCG@K (with a self-test) |
| `evaluation/results_collaborative.csv` | All experiment results |
| `notebooks/02_collaborative_filtering.ipynb` | Development: data exploration, KNN, SVD, interface tests, experiments, charts |