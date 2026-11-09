# Member 4 – MMR-Based Recommendation Diversity
## Project: CineXplain – Hybrid Movie Recommendation System

**Module:** Maximal Marginal Relevance (MMR)  
**Contributor:** Member 4

## 1. Introduction

CineXplain is a hybrid movie recommendation system that combines content-based and collaborative filtering techniques.

Traditional recommendation systems may recommend multiple movies with similar plots, genres or themes. To reduce repetitive recommendations, **Maximal Marginal Relevance (MMR)** is implemented as a re-ranking technique that balances relevance and diversity.

The objective is to generate personalized recommendations while maintaining variety.

## 2. Problem Statement

Selecting movies solely based on relevance scores can result in redundant recommendations.

The MMR module aims to:
- Preserve recommendation relevance.
- Reduce similarity among selected movies.
- Improve recommendation diversity.
- Balance personalization and variety.

## 3. Maximal Marginal Relevance

MMR is a greedy re-ranking algorithm that considers two factors:

- **Relevance:** How well a movie matches user preferences.
- **Diversity:** How different the movie is from already selected movies.

### Mathematical Formulation

\[
MMR(d)=\lambda Rel(d)-(1-\lambda)
\max_{s\in S}Sim(d,s)
\]

Where:

- \(d\): Candidate movie.
- \(S\): Already selected movies.
- \(Rel(d)\): Relevance score.
- \(Sim(d,s)\): Similarity between movies.
- \(\lambda\): Relevance-diversity trade-off parameter.

### Lambda Parameter

| Lambda | Behaviour |
|---|---|
| 1.0 | Pure relevance |
| 0.9 | Strong relevance preference |
| 0.7 | Relevance with diversity |
| 0.5 | Balanced relevance and diversity |
| 0.3 | Stronger diversity preference |
| 0.0 | Maximum diversity pressure |

Different lambda values are tested to identify a suitable balance.

## 4. Dataset and Preprocessing

**Dataset:** `movies_enriched.csv`

Movie metadata is used to calculate content similarity. Available features may include:

- Movie title and overview/plot.
- Genres and cast.
- Keywords and director.
- Tagline and release year.

Preprocessing involves:

1. Loading the dataset using Pandas.
2. Identifying available metadata columns.
3. Handling missing values.
4. Combining relevant attributes into textual representations.
5. Generating TF-IDF vectors.

## 5. Content Similarity Calculation

### TF-IDF

TF-IDF converts movie descriptions into numerical vectors by measuring the importance of terms.

\[
TFIDF(t,d)=TF(t,d)\times IDF(t)
\]

It gives greater importance to informative terms and reduces the influence of common terms.

### Cosine Similarity

Cosine similarity measures the similarity between movie vectors.

\[
Cosine(A,B)=\frac{A\cdot B}{\|A\|\|B\|}
\]

Values closer to 1 indicate greater similarity, while values closer to 0 indicate lower similarity.

The resulting pairwise similarity matrix is used by MMR.

## 6. MMR Implementation

The implementation is maintained in `src/mmr.py`.

The module performs the following operations:

1. **Dataset Loading:** Loads the enriched movie CSV using Pandas.
2. **Text Preparation:** Combines available movie metadata and handles missing values.
3. **Similarity Generation:** Uses TF-IDF and cosine similarity to generate pairwise movie similarities.
4. **Candidate Selection:** Receives candidate movie IDs and hybrid relevance scores.
5. **Greedy Re-ranking:** Iteratively selects movies based on their MMR scores.
6. **Final Output:** Returns a re-ranked list of diverse movie recommendations.

### Greedy Algorithm

```text
Initialize selected_movies = []

While selected_movies has fewer than K movies:

    For each remaining candidate:
        Calculate relevance.

        If selected_movies is empty:
            Diversity penalty = 0
        Else:
            Diversity penalty = maximum similarity
                               with selected movies

        Calculate MMR score.

    Select the candidate with the highest MMR score.
    Add it to selected_movies.

Return selected_movies.
```

The algorithm begins with the highest-relevance movie and subsequently balances relevance against similarity to previously selected movies.

## 7. Integration with Other Modules

MMR acts as a re-ranking component rather than an independent recommendation model.

- **Member 1 – Content-Based Filtering:** Provides movie representations and content similarity information.
- **Member 3 – Hybrid Recommendation:** Provides hybrid relevance scores by combining content-based and collaborative filtering signals.
- **Member 4 – MMR:** Uses both inputs to generate diverse recommendations.

### Overall Workflow

```text
Enriched Movie Dataset
          |
          v
Content Feature Extraction
          |
          v
TF-IDF / Similarity Matrix
          |
          +----------------------+
                                 |
Hybrid Recommendation Scores    |
          |                      |
          v                      |
Candidate Movie Selection       |
          |                      |
          +----------+-----------+
                     |
                     v
               MMR Re-ranking
                     |
                     v
          Diverse Recommendations
```

## 8. Evaluation Methodology

MMR is evaluated by comparing the original hybrid ranking with the MMR re-ranked list.

### 8.1 Intra-List Diversity (ILD)

Measures diversity among recommended movies.

\[
ILD(L)=1-\frac{2}{|L|(|L|-1)}
\sum_{i<j}Sim(i,j)
\]

Higher ILD indicates greater recommendation diversity.

### 8.2 Precision@K

Measures the proportion of relevant movies among the top K recommendations.

\[
Precision@K=\frac{\text{Relevant movies in top K}}{K}
\]

### 8.3 NDCG@K

Measures ranking quality by considering relevance and recommendation positions.

\[
NDCG@K=\frac{DCG@K}{IDCG@K}
\]

These metrics help evaluate whether diversity improvements affect recommendation relevance.

## 9. Experimental Setup

The following experiments are planned:

**Experiment 1 – Lambda Comparison**

Test lambda values: 0.3, 0.5, 0.7, 0.9 and 1.0.

**Experiment 2 – Hybrid vs MMR**

Compare original hybrid rankings with MMR re-ranked recommendations using ILD, Precision@K and NDCG@K.

**Experiment 3 – Recommendation List Size**

Test different K values to understand the effect of list length on diversity and relevance.

**Experiment 4 – Similarity Representation**

Compare TF-IDF-based similarity with sentence-embedding-based similarity, where available.

### Experimental Results

Actual results will be populated after executing the experiments.

| Lambda | ILD | Precision@K | NDCG@K |
|---|---|---|---|
| 0.3 | TBD | TBD | TBD |
| 0.5 | TBD | TBD | TBD |
| 0.7 | TBD | TBD | TBD |
| 0.9 | TBD | TBD | TBD |
| 1.0 | TBD | TBD | TBD |

### Hybrid vs MMR Comparison

| Metric | Hybrid Ranking | MMR Ranking |
|---|---|---|
| ILD | TBD | TBD |
| Precision@K | TBD | TBD |
| NDCG@K | TBD | TBD |

**Observations to record:**
- Change in recommendation diversity.
- Effect of lambda on relevance.
- Best relevance-diversity balance.
- Effect of similarity representation.
- Effect of recommendation list size.

## 10. Advantages and Limitations

### Advantages

- Reduces repetitive recommendations.
- Improves movie variety.
- Balances relevance and diversity.
- Integrates with hybrid recommendation scores.
- Does not require training a separate recommendation model.
- Offers controllable diversity through lambda.

### Limitations

- Depends on the quality of hybrid relevance scores.
- Requires sufficient movie metadata.
- TF-IDF may not capture deeper semantic relationships.
- Lambda selection influences recommendation quality.
- Pairwise similarity computation can be expensive for large candidate pools.
- Higher diversity does not automatically guarantee relevance.

## 11. Individual Contribution – Member 4

The individual contribution includes:

1. Studying MMR and its application to recommendation systems.
2. Designing the relevance-diversity re-ranking approach.
3. Implementing the algorithm in `src/mmr.py`.
4. Using enriched movie metadata for content similarity.
5. Implementing TF-IDF vectorization and cosine similarity.
6. Developing the greedy movie selection procedure.
7. Planning lambda-based experiments.
8. Defining ILD, Precision@K and NDCG@K evaluation metrics.
9. Developing evaluation utilities in `evaluation/mmr_metrics.py`.
10. Working on integration with the hybrid recommendation pipeline.

## 12. Conclusion

MMR improves CineXplain by introducing diversity into hybrid movie recommendations.

By combining hybrid relevance scores with movie content similarity, it reduces redundant recommendations while maintaining personalization.

It complements the content-based and collaborative filtering modules and aims to provide users with recommendations that are relevant, diverse and engaging.

---

**Project:** CineXplain – Hybrid Movie Recommendation System  
**Module:** MMR-Based Recommendation Diversity  
**Contributor:** Member 4