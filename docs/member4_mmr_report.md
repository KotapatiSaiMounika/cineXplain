# CineXplain – MMR-Based Recommendation Diversity

**Project:** CineXplain – Hybrid Movie Recommendation System
**Module:** Maximal Marginal Relevance (MMR) re-ranking
**Contributor:** Satwika (Member 4)
**Files:** `src/mmr.py` · `evaluation/mmr_metrics.py` · data: `movies_enriched.csv`

---

## 1. Overview

CineXplain combines content-based and collaborative filtering into a hybrid recommender. A list sorted purely by hybrid score often contains near-duplicates (e.g. ten similar space-survival films).

This module re-ranks the hybrid model's top candidates with **Maximal Marginal Relevance (MMR)** so the final list stays relevant to the user but is not repetitive.

## 2. Problem Statement

Relevance-only ranking causes redundancy, because the signals that make movies relevant (same genre, franchise, director, keywords) also make them similar. The MMR module aims to:

- Preserve relevance.
- Reduce similarity among selected movies.
- Improve diversity.
- Let us tune the balance with one parameter.

## 3. Approach: Greedy MMR Re-ranking

**Algorithm used: greedy, iterative selection.** The final list is built one movie at a time. At each step, every remaining candidate is scored against the movies already selected, and the highest-scoring candidate is added. A candidate's score changes after every pick, because the selected set changes.

### Formula

```
MMR(d) = λ · Rel(d) − (1 − λ) · max_{s ∈ S} Sim(d, s)
```

| Symbol | Meaning |
|---|---|
| `d` | Candidate movie |
| `S` | Movies already selected |
| `Rel(d)` | Hybrid relevance score (normalised to 0–1) |
| `Sim(d, s)` | Cosine similarity of TF-IDF vectors |
| `λ` | Relevance–diversity trade-off |

The `max` term penalises a candidate if it is close to even one selected movie.

### Greedy algorithm

```text
Input : candidates C with Rel scores, similarity matrix M, list size K, lambda
Output: re-ranked list S

S ← []
while |S| < K and candidates remain:
    for each remaining candidate d:
        if S is empty:  penalty ← 0
        else:           penalty ← max over s in S of M[d][s]
        mmr[d] ← λ · Rel(d) − (1 − λ) · penalty
    pick d* with the highest mmr score
    add d* to S, remove it from candidates
return S
```

The first pick has no penalty, so the algorithm always starts with the highest-relevance movie. It is greedy: each step takes the best local choice and never revisits earlier picks.

### Lambda

| λ | Behaviour |
|---|---|
| 1.0 | Pure relevance (original hybrid ranking) |
| 0.9 | Strong relevance preference |
| 0.7 | Relevance with diversity |
| 0.5 | Balanced |
| 0.3 | Stronger diversity preference |
| 0.0 | Maximum diversity (avoid; first pick has no relevance signal) |

### Example (λ = 0.7)

| Movie | Rel | Sim to A |
|---|---|---|
| A – Space Survival 1 | 0.90 | – |
| B – Space Survival 2 | 0.85 | 0.90 |
| C – Heist Comedy | 0.70 | 0.20 |

A is picked first. Then `MMR(B) = 0.7×0.85 − 0.3×0.90 = 0.325` and `MMR(C) = 0.7×0.70 − 0.3×0.20 = 0.430`, so **C beats B** and the list is `[A, C]` instead of `[A, B]`.

## 4. Data and Similarity

**Dataset:** `movies_enriched.csv`, loaded with Pandas.

1. Detect which metadata columns exist (title, overview, genres, cast, director, keywords, tagline, year).
2. Fill missing values with empty strings.
3. Flatten list-like fields and join multi-word names (e.g. `science_fiction`).
4. Combine the fields into one text document per movie.
5. Build **TF-IDF** vectors, which weight distinctive terms over common ones: `TFIDF(t,d) = TF(t,d) × IDF(t)`.
6. Compute **cosine similarity**: `Cosine(A,B) = (A·B) / (‖A‖‖B‖)`. Values lie in 0–1, the same scale as normalised relevance.

The similarity matrix is computed only for the candidate pool (e.g. top 50–100 hybrid results), not the whole catalogue.

## 5. Implementation (`src/mmr.py`)

1. Load the dataset.
2. Prepare text from metadata.
3. Generate TF-IDF and cosine similarities.
4. Receive candidate movie IDs and hybrid scores.
5. Run the greedy MMR loop.
6. Return the diverse re-ranked list.

Complexity: O(N·K²) for the loop over N candidates and K picks (O(N·K) if each candidate's running max similarity is cached); O(N²) memory for the similarity matrix.

## 6. Integration

MMR is a re-ranking layer, not a standalone recommender.

- **Member 1 – Content-Based Filtering:** content representations and similarity.
- **Member 3 – Hybrid Recommendation:** relevance scores.
- **Member 4 – MMR:** combines both into the final diverse list.

```text
Enriched Dataset → TF-IDF / Similarity Matrix ──┐
Hybrid Scores → Candidate Selection (top-N) ────┤
                                                ▼
                                          MMR Re-ranking
                                                ▼
                                    Diverse Top-K Recommendations
```

## 7. Evaluation

The original hybrid ranking is compared with the MMR ranking on the same users and candidate pool. Utilities are in `evaluation/mmr_metrics.py`.

- **Intra-List Diversity (ILD):** `1 − [2 / (|L|(|L|−1))] · Σ_{i<j} Sim(i,j)`. Higher means more diverse.
- **Precision@K:** relevant movies in top K, divided by K.
- **NDCG@K:** `DCG@K / IDCG@K`, which rewards relevant movies placed higher.

A small drop in Precision/NDCG alongside a rise in ILD is the expected trade-off.

## 8. Experiments

1. **Lambda comparison:** λ = 0.3, 0.5, 0.7, 0.9, 1.0.
2. **Hybrid vs MMR:** ILD, Precision@K, NDCG@K.
3. **List size:** different values of K.
4. **Similarity representation:** TF-IDF vs sentence embeddings (if available).

## 9. Advantages and Limitations

**Advantages**
- Reduces repetitive recommendations and improves variety.
- No extra model training needed.
- Diversity is controllable through λ.
- Works with any hybrid scoring method.

**Limitations**
- Depends on the quality of hybrid relevance scores and metadata.
- TF-IDF misses deeper semantic relationships.
- Results are sensitive to λ.
- Pairwise similarity is quadratic in pool size.
- Greedy selection does not guarantee the globally optimal diverse set.

## 10. Conclusion

MMR adds diversity to CineXplain's hybrid recommendations using a greedy re-ranking that trades relevance against similarity to already-selected movies. It needs no extra training and is tuned with a single parameter, λ.

---

**Project:** CineXplain · **Module:** MMR-Based Recommendation Diversity · **Contributor:** Satwika (Member 4)
