# Member 1: Content-Based Recommendation

**Ownership:** movie metadata processing and content similarity.
**Deliverable:** [`src/content_recommender.py`](../src/content_recommender.py)
**Evaluation:** [`evaluation/content_metrics.py`](../evaluation/content_metrics.py), results in [`evaluation/results_content.csv`](../evaluation/results_content.csv)

## What it does

Recommends movies similar to what a user already liked, using only movie *content* (genres, cast, plot, year, title), not other users' ratings.

1. Loads and cleans the movie table (year extraction, title cleaning, missing genres).
2. Builds one text document per movie.
3. Represents each movie as a vector in two ways: **TF-IDF** and **sentence embeddings**.
4. Builds a user taste vector from the movies they liked.
5. Ranks all unseen movies by **cosine similarity** to that taste vector and returns the Top-K.

## How to run

### 1. Setup
See the [root README](../README.md#setup) for environment and data setup.

### 2. Enrich the metadata (one time)

MovieLens has only `movieId, title, genres`. There is no plot or cast. `scripts/enrich_metadata.py` fetches them from the free [TMDB API](https://www.themoviedb.org/settings/api) using `links.csv`.

```bash
export TMDB_API_KEY=your_key                      # Windows PowerShell: $env:TMDB_API_KEY="your_key"
python scripts/enrich_metadata.py --limit 50      # test on 50 movies first
python scripts/enrich_metadata.py                 # then everything (resumable)
```

Output: `data/movies_enriched.csv`

| Column | Meaning |
|---|---|
| `movieId` | MovieLens id |
| `overview` | Plot summary |
| `cast` | Top 8 actors, separated by `\|` |
| `collection` | Franchise name (e.g. "Toy Story Collection"), blank if none |

`content_recommender.py` loads this file automatically if it exists. Without it, only title and genres are used and results are much weaker (for example, sequels end up with identical vectors). The file is gitignored, so every teammate must generate it or receive it directly.

### 3. Demo

```bash
python src/content_recommender.py --movie "Toy Story" --method both --k 10
```

Options: `--method tfidf|embeddings|both`, `--k N`. The first embeddings run takes a few minutes on CPU, then the result is cached in `data/cache/`.

Example (embeddings, seed *Toy Story*):

| Title | Score |
|---|---|
| Toy Story 2 (1999) | 0.867 |
| Toy Story 3 (2010) | 0.802 |
| Toy, The (1982) | 0.688 |
| Toy Story Toons: Partysaurus Rex (2012) | 0.649 |

### 4. Evaluate

```bash
python evaluation/content_metrics.py --k 10 --users 600
```

## Design

### Text representation

| | TF-IDF | Sentence embeddings |
|---|---|---|
| Model | `TfidfVectorizer` (English stop words, sublinear TF) | `all-MiniLM-L6-v2` (384-dim, normalized) |
| Genres | Repeated 2x so they weigh more | Written as plain text ("Genres: Comedy, Fantasy") |
| Cast | Top 5, each name glued into one token (`tomhanks`) | Top 5, written as plain names |
| Plot | Cleaned words | Full sentence text |
| Extras | Decade token (`decade1990`), cleaned title | Title |

TF-IDF gets engineered tokens because it counts exact words. Embeddings get natural sentences because the model understands normal English.

### User taste profile

- A single `movieId` means the taste is that movie.
- A `{movieId: rating}` dict means taste is the **rating-weighted average** of the vectors of movies rated ≥ 4. If the user has none, all rated movies are used.
- Movies the user has already rated are excluded from results.

### Interface for the hybrid model (Member 3)

```python
from src.content_recommender import ContentRecommender, load_movies

rec = ContentRecommender(load_movies())
rec.fit_tfidf()
rec.fit_embeddings()

scores = rec.get_content_scores({1: 5.0, 296: 4.5, 50: 2.0},
                                method="embeddings", top_k=None)
# -> {movieId: cosine_similarity, ...} best first
```

- `top_k=None` returns scores for **all** unseen movies, which the hybrid needs.
- An empty or unknown profile returns `{}`, so the hybrid's cold-start logic handles it.
- Other helpers: `recommend()` (readable table), `compare_methods()` (both lists plus Jaccard overlap), `search_title()`.

## Evaluation

**Protocol.** Per user (at least 20 ratings), hide a random 20% of ratings and build the profile from the other 80%. Recommend the Top-10 unseen movies. A recommendation is a hit if the user rated it ≥ 4 in the hidden set. Fixed seed 42. Users with no liked movie in the train or test portion are skipped.

**Metrics.**
- **Precision@10:** hits / 10
- **Recall@10:** hits / number of liked hidden movies
- **GenreMatch@10:** share of recommendations that share at least one of the user's top-3 liked genres (a sanity check, not a quality score)
- **Random baseline** to show the model beats chance

**Results (2,952 users, K = 10; a random sample of 3,000 users with at least 20 ratings):**

| Method | Precision@10 | Recall@10 | GenreMatch@10 |
|---|---|---|---|
| Random | 0.0006 | 0.0003 | 0.625 |
| TF-IDF | 0.0033 | 0.0067 | 0.984 |
| Sentence embeddings | 0.0013 | 0.0030 | 0.814 |


### TF-IDF vs sentence embeddings

Both content-based methods beat the random baseline, but TF-IDF is clearly stronger in this setup: Precision@10 of 0.0033 vs 0.0013 for embeddings and 0.0006 for random (roughly 97 vs 38 vs 18 hits across 2,952 users). TF-IDF's representation weights genres, cast names and franchise or title tokens, which line up well with a taste profile built from liked movies. Sentence embeddings capture plot semantics and give more thematically varied lists (GenreMatch 0.81 vs 0.98), but averaging many liked movies into one embedding appears to dilute the signal, so they recover fewer of the hidden liked movies. Smaller runs (195 and 588 users) showed the order flipping, which was noise from too few hits. All results come from a single 80/20 split with seed 42, so they should be read as indicative. Top-10 overlap between the two methods is low to moderate (Jaccard about 0.2 to 0.5 in the demos), so they surface different movies, which can still be useful in a hybrid. Absolute precision is low for both, as expected for content-only recommendation with no popularity or collaborative signal; this motivates the hybrid model.

## Limitations

- **Low absolute precision.** Expected for content-only recommendation: it uses no popularity or collaborative signal, and there are only a few hidden liked movies per user. This is the motivation for the hybrid model.
- **Name and title matching.** Shared words can dominate. For example, *Titanic* also surfaces *Rose Red* and *The Rose* because the plot mentions a character called Rose.
- **TF-IDF favors rare words,** which can surface obscure movies that few users have rated.
- **Metadata gaps.** Some movies have no TMDB overview or cast, so they fall back to title and genres.
- **Single split and seed.** Results come from one train/test split; repeated splits would give error bars.
- **GenreMatch is inflated for TF-IDF** because genres are explicitly boosted in its text.

## Files

| File | Purpose |
|---|---|
| `src/content_recommender.py` | Cleaning, vectorization, similarity, Top-K, public API |
| `scripts/enrich_metadata.py` | Fetches plot, cast and collection from TMDB |
| `evaluation/content_metrics.py` | Train/test evaluation with Precision, Recall, GenreMatch and a random baseline |
| `evaluation/results_content.csv` | Latest evaluation results |
| `notebooks/eda.ipynb` | Metadata exploration: missing data, genre and year distributions, TMDB coverage, ratings overview |
| `notebooks/demo.ipynb` | Demo: TF-IDF vs embeddings side by side, user-profile recommendations, evaluation chart |