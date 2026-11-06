"""
CineXplain - evaluation metrics for the collaborative filtering module.

Rating prediction : RMSE, MAE
Ranking           : Precision@K, Recall@K, NDCG@K (binary relevance)

"Relevant" movie = a movie the user rated >= relevant_threshold in the TEST set.
Candidates       = every movie the user has NOT rated in the TRAIN set.
"""
import numpy as np
from scipy.sparse import csr_matrix


def rmse_mae(pred, true):
    """Return (RMSE, MAE). Predictions are clipped to the valid rating range 0.5-5.0."""
    pred = np.clip(np.asarray(pred, dtype=np.float64), 0.5, 5.0)
    true = np.asarray(true, dtype=np.float64)
    return float(np.sqrt(np.mean((pred - true) ** 2))), float(np.mean(np.abs(pred - true)))


def build_relevance_matrix(test_user_idx, test_movie_idx, test_ratings, shape, threshold=4.0):
    """Sparse 0/1 matrix (users x movies): 1 where the user rated the movie >= threshold in the test set."""
    keep = np.asarray(test_ratings) >= threshold
    rel = csr_matrix((np.ones(keep.sum(), dtype=np.float32),
                      (np.asarray(test_user_idx)[keep], np.asarray(test_movie_idx)[keep])), shape=shape)
    rel.data[:] = 1
    return rel


def ranking_metrics(score_block_fn, train_csr, rel_csr, ks=(10, 20), block=1000):
    """
    Precision@K, Recall@K and NDCG@K for every user that has at least one relevant test movie.

    score_block_fn(user_indices) -> dense array (len(user_indices), n_movies); higher = better.
                                    -inf marks "not a candidate".
    train_csr : users x movies train ratings (already-rated movies are never recommended)
    rel_csr   : users x movies relevance matrix from build_relevance_matrix

    Returns (evaluated_user_indices, {"Precision@10": per-user array, ...}).
    """
    n_rel = np.diff(rel_csr.indptr)
    users = np.where(n_rel > 0)[0]
    maxk = max(ks)
    disc = 1.0 / np.log2(np.arange(2, maxk + 2))
    idcg_cum = np.cumsum(disc)
    out = {f"{m}@{k}": np.zeros(len(users)) for m in ("Precision", "Recall", "NDCG") for k in ks}
    for a in range(0, len(users), block):
        ub = users[a:a + block]
        sc = np.array(score_block_fn(ub), dtype=np.float32)
        rr, cc = train_csr[ub].nonzero()
        sc[rr, cc] = -np.inf                                        # never recommend already-rated movies
        top = np.argpartition(-sc, maxk, axis=1)[:, :maxk]
        ts = np.take_along_axis(sc, top, 1)
        order = np.argsort(-ts, axis=1)
        top = np.take_along_axis(top, order, 1)
        ts = np.take_along_axis(ts, order, 1)
        rel = rel_csr[ub].toarray() > 0
        hits = np.take_along_axis(rel, top, 1) & np.isfinite(ts)    # a -inf slot is not a real recommendation
        nr = n_rel[ub]
        for k in ks:
            h = hits[:, :k]
            out[f"Precision@{k}"][a:a + len(ub)] = h.sum(1) / k
            out[f"Recall@{k}"][a:a + len(ub)] = h.sum(1) / nr
            out[f"NDCG@{k}"][a:a + len(ub)] = (h * disc[:k]).sum(1) / idcg_cum[np.minimum(nr, k) - 1]
    return users, out


def summarise(per_user):
    """Average each per-user metric array."""
    return {k: float(v.mean()) for k, v in per_user.items()}


if __name__ == "__main__":
    # Self-test on a tiny example that can be checked by hand.
    # 3 users x 6 movies. Movie 5 is already rated by user 0 in train (must be excluded even though it scores highest).
    train = csr_matrix(np.array([[0, 0, 0, 0, 0, 5], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=np.float32))
    rel = csr_matrix(np.array([[0, 1, 0, 1, 0, 0],      # user 0 relevant: movies 1, 3
                               [1, 0, 0, 0, 0, 0],      # user 1 relevant: movie 0
                               [0, 0, 1, 0, 1, 0]],     # user 2 relevant: movies 2, 4
                              dtype=np.float32))
    scores = np.array([[0.1, 0.9, 0.3, 0.8, 0.2, 5.0],   # user 0: ranks 1,3 on top (movie 5 excluded) -> perfect
                       [0.2, 0.9, 0.5, 0.1, 0.0, 0.3],   # user 1: top-2 = movies 1,2 -> miss
                       [0.9, 0.1, 0.8, 0.0, 0.2, 0.1]])  # user 2: top-2 = movies 0,2 -> one hit at rank 2
    users, pu = ranking_metrics(lambda ub: scores[ub], train, rel, ks=(2,), block=2)
    assert list(users) == [0, 1, 2]
    assert np.allclose(pu["Precision@2"], [1.0, 0.0, 0.5])
    assert np.allclose(pu["Recall@2"], [1.0, 0.0, 0.5])
    expected_ndcg_u2 = (1 / np.log2(3)) / (1 + 1 / np.log2(3))            # 0.3869
    assert np.allclose(pu["NDCG@2"], [1.0, 0.0, expected_ndcg_u2]), pu["NDCG@2"]
    rm, ma = rmse_mae([3.0, 4.0, 6.0], [3.0, 5.0, 5.0])                   # 6.0 is clipped to 5.0
    assert abs(ma - 1 / 3) < 1e-9 and abs(rm - np.sqrt(1 / 3)) < 1e-9
    print("collaborative_metrics self-test passed")