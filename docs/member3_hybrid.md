
# CineXplain – Hybrid Recommender Report

## 1. Objective
Combine content-based and collaborative filtering
using adaptive weighting.

## 2. Implemented Components
- Adaptive Cold-Start Gate
- Content and CF score integration
- Score normalization
- Missing prediction handling
- Hybrid recommendation ranking
- Precision@K
- Recall@K
- NDCG@K
- Recommendation coverage

## 3. Cold-Start Strategy

alpha = k / (k + num_interactions)

Content weight = alpha
CF weight = 1 - alpha

New users receive greater content weight.
Experienced users receive greater CF weight.

## 4. Dataset
MovieLens (about 138k users). Temporal split: each user's most recent rating is
held out for testing.
- Training ratings: 1,986,177
- Testing ratings: 138,493 (one per user)

## 5. Tuning
`evaluation/tune_hybrid_k.py` tested k = 10, 25, 50, 68, 100, 150 and chose
k = 68 based on recall. At 68 interactions, content and CF are weighted 50/50.

## 6. Results (Recall@10, one held-out movie per user)
| Run | Precision@10 | Recall@10 |
|---|---|---|
| Hybrid, all users | ~0.005 | ~0.049 |
| Hybrid, temporal split | ~0.0075 | ~0.075 |

Numbers are low because each user has only one held-out test movie and the
catalogue is large. See `reports/` for the CSVs.

## 7. Integration
The hybrid scores feed MMR (`src/mmr.py`) and then the CSP filter
(`src/constraints.py`). The Streamlit app (`app.py`, `src/streamlit_engine.py`)
runs the whole pipeline, with constraints applied before MMR.