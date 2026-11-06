"""
CineXplain - Collaborative Filtering module (Member 2)

Two algorithms, built only from TRAIN ratings (no leakage):
  1. Item-Item KNN : adjusted-cosine movie similarity, top-K neighbours per movie
  2. SVD           : biased matrix factorization  rating ~ mu + b_user + b_movie + p_user . q_movie
                     trained with full-batch gradient descent (Adam) + L2 regularisation

Main entry point for the hybrid recommender (Member 3):

    from src.collaborative_recommender import get_cf_scores
    scores = get_cf_scores(user_id=5, method="knn", top_k=100)   # {movieId: score in [0, 1]}

The first call builds / loads the models (about 1 minute the very first time, a few seconds afterwards,
because the models are cached in data/cache/).

Working set (agreed defaults, see constants below): ML-32M restricted to movies with >= 50 ratings and a
random sample of 30,000 users; 80/20 per-user random train/test split, seed 42.
"""
import os
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# ----------------------------------------------------------------------------- configuration
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")

MIN_MOVIE_RATINGS = 50     # keep movies with at least this many ratings (full data)
N_DEV_USERS = 30000        # users in the working set
TEST_RATIO = 0.2           # per-user test share
SEED = 42
RELEVANT_THRESHOLD = 4.0   # test rating >= this counts as "relevant" in ranking metrics
LIKE_THRESHOLD = 4.0       # train rating >= this counts as "liked" when scoring with KNN

K_NEIGHBOURS = 50          # Item-KNN neighbours per movie
N_FACTORS = 50             # SVD latent factors
SVD_REG = 0.05
SVD_LR = 0.02
SVD_EPOCHS = 40
POP_WEIGHT = 0.5           # SVD ranking score = predicted rating + POP_WEIGHT * log(popularity)
MIN_HISTORY = 5            # fewer usable ratings than this -> popularity fallback (cold start)

_MODEL = None              # lazily built default model, see get_model()


# ----------------------------------------------------------------------------- data
def load_ratings(data_dir=None):
    """Load MovieLens ratings.csv and movies.csv. Returns (ratings, movies)."""
    data_dir = data_dir or DATA_DIR
    ratings = pd.read_csv(os.path.join(data_dir, "ratings.csv"),
                          usecols=["userId", "movieId", "rating", "timestamp"],
                          dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int32"})
    movies = pd.read_csv(os.path.join(data_dir, "movies.csv"))
    return ratings, movies


def split_train_test(ratings, min_movie_ratings=MIN_MOVIE_RATINGS, n_users=N_DEV_USERS,
                     test_ratio=TEST_RATIO, seed=SEED):
    """Build the working set and the per-user random train/test split. Returns (train, test)."""
    mc = ratings.groupby("movieId").size()
    keep_movies = mc.index[mc >= min_movie_ratings]
    df = ratings[ratings.movieId.isin(keep_movies)]
    uc = df.groupby("userId").size()
    eligible = uc.index[uc >= 20]
    rng = np.random.RandomState(seed)
    dev_users = rng.choice(eligible, size=n_users, replace=False)
    df = df[df.userId.isin(dev_users)].copy()
    df["_r"] = rng.random_sample(len(df))
    df["_rank"] = df.groupby("userId")["_r"].rank(method="first") - 1
    n_user = df.groupby("userId")["rating"].transform("size")
    n_test = np.maximum(1, np.round(n_user * test_ratio)).astype(int)
    is_test = df["_rank"] < n_test
    train = df[~is_test].drop(columns=["_r", "_rank"]).reset_index(drop=True)
    test = df[is_test].drop(columns=["_r", "_rank"]).reset_index(drop=True)
    return train, test


def prepare_interactions(train, test, movies):
    """Build the user-item matrix (TRAIN only) and all lookup tables. Returns a SimpleNamespace."""
    user_ids = np.sort(train.userId.unique())
    movie_ids = np.sort(train.movieId.unique())
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    movie_to_idx = {m: i for i, m in enumerate(movie_ids)}
    R = csr_matrix((train.rating.values.astype(np.float32),
                    (train.userId.map(user_to_idx).values, train.movieId.map(movie_to_idx).values)),
                   shape=(len(user_ids), len(movie_ids)))
    n_u = np.diff(R.indptr).astype(np.float32)
    n_i = np.asarray((R > 0).sum(0)).ravel().astype(np.float32)
    te = test[test.movieId.isin(movie_to_idx)]            # a test movie never seen in train cannot be scored
    return SimpleNamespace(
        train=train, test=test, R=R, user_ids=user_ids, movie_ids=movie_ids,
        user_to_idx=user_to_idx, movie_to_idx=movie_to_idx,
        n_users=R.shape[0], n_movies=R.shape[1], n_u=n_u, n_i=n_i, log_pop=np.log(n_i),
        user_mean=np.asarray(R.sum(1)).ravel() / n_u, mu=float(R.data.mean()),
        te_u=te.userId.map(user_to_idx).values.astype(np.int32),
        te_i=te.movieId.map(movie_to_idx).values.astype(np.int32),
        te_r=te.rating.values.astype(np.float32),
        titles=movies.set_index("movieId").title)


# ----------------------------------------------------------------------------- Item-Item KNN
def build_item_similarity(R, user_mean, K=K_NEIGHBOURS):
    """Adjusted-cosine item-item similarity (user mean removed). Returns top-K neighbour indices and similarities
    for every movie, computed block by block so memory stays small."""
    Rc = R.copy()
    Rc.data = Rc.data - np.repeat(user_mean, np.diff(Rc.indptr)).astype(np.float32)
    X = Rc.T.tocsr()                                              # movies x users
    norms = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
    norms[norms == 0] = 1
    X = csr_matrix(X.multiply(1 / norms[:, None]), dtype=np.float32)
    XT = X.T.tocsc()
    n = X.shape[0]
    idx = np.zeros((n, K), dtype=np.int32)
    sim = np.zeros((n, K), dtype=np.float32)
    for s in range(0, n, 1000):
        blk = (X[s:s + 1000] @ XT).toarray()
        for j in range(blk.shape[0]):
            blk[j, s + j] = -1                                    # a movie is not its own neighbour
        top = np.argpartition(-blk, K, axis=1)[:, :K]
        ts = np.take_along_axis(blk, top, 1)
        order = np.argsort(-ts, 1)
        idx[s:s + 1000] = np.take_along_axis(top, order, 1)
        sim[s:s + 1000] = np.take_along_axis(ts, order, 1)
    return idx, sim


def neighbour_matrix(idx, sim, k=None):
    """Sparse movies x movies matrix holding each movie's top-k POSITIVE similarities."""
    k = k or idx.shape[1]
    w = np.clip(sim[:, :k], 0, None)
    return csr_matrix((w.ravel(), (np.repeat(np.arange(idx.shape[0]), k), idx[:, :k].ravel())),
                      shape=(idx.shape[0],) * 2)


def _knn_scores(model, idx, r):
    """Score of a movie = summed similarity to the movies the user liked (rating >= LIKE_THRESHOLD)."""
    n = model.n_movies
    ind = np.zeros(n, dtype=np.float32); ind[idx] = 1
    liked = np.zeros(n, dtype=np.float32); liked[idx[r >= LIKE_THRESHOLD]] = 1
    evidence = model.N @ ind                                     # similarity weight of the user's rated neighbours
    score = (model.N @ liked).astype(np.float64)
    return np.where(evidence > 0, score, -np.inf)                # no neighbour evidence -> not a candidate


# ----------------------------------------------------------------------------- SVD
def _dot_rows(P, Q, u, i, chunk=400000):
    out = np.empty(len(u), np.float32)
    for s in range(0, len(u), chunk):
        out[s:s + chunk] = (P[u[s:s + chunk]] * Q[i[s:s + chunk]]).sum(1)
    return out


def train_svd(R, k=N_FACTORS, reg=SVD_REG, lr=SVD_LR, epochs=SVD_EPOCHS, seed=SEED, verbose=False):
    """Biased matrix factorization trained on the sparse train matrix R (users x movies).
    Full-batch gradient descent with the Adam optimiser and L2 regularisation weighted by rating counts.
    Returns (P, Q, bu, bi, mu)."""
    nU, nI = R.shape
    tr_u = np.repeat(np.arange(nU), np.diff(R.indptr)).astype(np.int32)
    tr_i = R.indices.astype(np.int32)
    tr_r = R.data.astype(np.float32)
    mu = float(tr_r.mean())
    n_u = np.diff(R.indptr).astype(np.float32)
    n_i = np.asarray((R > 0).sum(0)).ravel().astype(np.float32)
    rng = np.random.RandomState(seed)
    P = (0.1 * rng.randn(nU, k)).astype(np.float32)               # user latent factors
    Q = (0.1 * rng.randn(nI, k)).astype(np.float32)               # movie latent factors
    bu = np.zeros(nU, np.float32); bi = np.zeros(nI, np.float32)
    params = [P, Q, bu, bi]
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    scale = 1000 / len(tr_r)
    for ep in range(1, epochs + 1):
        pred = mu + bu[tr_u] + bi[tr_i] + _dot_rows(P, Q, tr_u, tr_i)
        E = csr_matrix((tr_r - pred, R.indices, R.indptr), shape=R.shape)   # errors laid out like R
        grads = [(-(E @ Q) + reg * n_u[:, None] * P) * scale,
                 (-(E.T.tocsr() @ P) + reg * n_i[:, None] * Q) * scale,
                 (-np.asarray(E.sum(1)).ravel() + reg * n_u * bu) * scale,
                 (-np.asarray(E.sum(0)).ravel() + reg * n_i * bi) * scale]
        for j, (p, g) in enumerate(zip(params, grads)):
            m[j] = b1 * m[j] + (1 - b1) * g
            v[j] = b2 * v[j] + (1 - b2) * g * g
            p -= lr * (m[j] / (1 - b1 ** ep)) / (np.sqrt(v[j] / (1 - b2 ** ep)) + eps)
        if verbose and (ep % 10 == 0 or ep == 1):
            rmse = float(np.sqrt(np.mean((np.clip(pred, 0.5, 5) - tr_r) ** 2)))
            print(f"  [svd] epoch {ep:3d} train RMSE {rmse:.4f}")
    return P, Q, bu, bi, mu


def predict_svd(user_id, movie_id, model=None):
    """Predicted star rating (0.5-5). Unknown user and/or movie fall back to the best available average."""
    model = model or get_model()
    u = model.user_to_idx.get(user_id); i = model.movie_to_idx.get(movie_id)
    pred = model.mu
    if u is not None: pred += model.bu[u]
    if i is not None: pred += model.bi[i]
    if u is not None and i is not None: pred += float(model.P[u] @ model.Q[i])
    return float(np.clip(pred, 0.5, 5))


def _svd_scores(model, idx, r, u):
    if u is not None:                                            # user seen in training: learned factors
        p = model.P[u]
    else:                                                        # new user: "fold in" by ridge regression
        k = model.Q.shape[1]
        A = np.hstack([model.Q[idx], np.ones((len(idx), 1), dtype=np.float32)])
        y = r - model.mu - model.bi[idx]
        x = np.linalg.solve(A.T @ A + SVD_REG * len(idx) * np.eye(k + 1), A.T @ y)
        p = x[:k].astype(np.float32)
    return (model.bi + model.Q @ p + POP_WEIGHT * model.log_pop).astype(np.float64)


# ----------------------------------------------------------------------------- model build / cache
def get_model(force_rebuild=False, use_cache=True, verbose=True):
    """Build (or load from data/cache) the default model: data, KNN neighbours and SVD factors."""
    global _MODEL
    if _MODEL is not None and not force_rebuild:
        return _MODEL
    t0 = time.time()
    os.makedirs(CACHE_DIR, exist_ok=True)
    train_p, test_p = os.path.join(CACHE_DIR, "cf_train.pkl"), os.path.join(CACHE_DIR, "cf_test.pkl")
    movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    if use_cache and os.path.exists(train_p) and os.path.exists(test_p):
        train, test = pd.read_pickle(train_p), pd.read_pickle(test_p)
        if verbose: print("[cf] loaded cached train/test split")
    else:
        if verbose: print("[cf] building working set from ratings.csv ...")
        ratings, _ = load_ratings()
        train, test = split_train_test(ratings)
        train.to_pickle(train_p); test.to_pickle(test_p)
    d = prepare_interactions(train, test, movies)

    model_p = os.path.join(CACHE_DIR, f"cf_model_k{K_NEIGHBOURS}_f{N_FACTORS}.npz")
    loaded = False
    if use_cache and os.path.exists(model_p):
        z = np.load(model_p)
        if z["P"].shape == (d.n_users, N_FACTORS) and z["Q"].shape == (d.n_movies, N_FACTORS) and z["idx"].shape == (d.n_movies, K_NEIGHBOURS):
            idx, sim, P, Q, bu, bi = z["idx"], z["sim"], z["P"], z["Q"], z["bu"], z["bi"]
            loaded = True
            if verbose: print("[cf] loaded cached KNN + SVD models")
    if not loaded:
        if verbose: print("[cf] building item similarity ...")
        idx, sim = build_item_similarity(d.R, d.user_mean, K_NEIGHBOURS)
        if verbose: print("[cf] training SVD ...")
        P, Q, bu, bi, _ = train_svd(d.R, verbose=verbose)
        np.savez(model_p, idx=idx, sim=sim, P=P, Q=Q, bu=bu, bi=bi)
    _MODEL = SimpleNamespace(**vars(d), nbr_idx=idx, nbr_sim=sim, N=neighbour_matrix(idx, sim, K_NEIGHBOURS),
                             P=P, Q=Q, bu=bu, bi=bi)
    if verbose: print(f"[cf] ready in {time.time() - t0:.1f}s | {d.n_users} users, {d.n_movies} movies")
    return _MODEL


# ----------------------------------------------------------------------------- recommendation interface
def _history_vectors(model, user_id, rating_history):
    """Return (movie_indices, ratings, trained_user_index_or_None) for this request."""
    if rating_history:
        pairs = [(model.movie_to_idx[m], float(r)) for m, r in rating_history.items() if m in model.movie_to_idx]
        return (np.array([p[0] for p in pairs], dtype=np.int64),
                np.array([p[1] for p in pairs], dtype=np.float32), None)
    if user_id is not None and user_id in model.user_to_idx:
        u = model.user_to_idx[user_id]
        s, e = model.R.indptr[u], model.R.indptr[u + 1]
        return model.R.indices[s:e], model.R.data[s:e], u
    return np.array([], dtype=np.int64), np.array([], dtype=np.float32), None


def get_cf_scores(user_id=None, method="knn", top_k=100, rating_history=None, normalize=True, model=None):
    """
    Collaborative-filtering candidate scores for the hybrid recommender.

    Parameters
    ----------
    user_id        : MovieLens userId (optional if rating_history is given)
    method         : 'knn' (item-item KNN) or 'svd' (matrix factorization)
    top_k          : number of candidate movies to return
    rating_history : optional {movieId: rating}. If given it is used instead of the stored history, so users
                     who are NOT in the training data can still get recommendations.
    normalize      : True -> scores min-max scaled to [0, 1] across the returned candidates (best = 1.0)

    Returns
    -------
    dict {movieId: score}, best first. Already-rated movies are excluded; unknown movieIds in rating_history are
    ignored. Users with fewer than MIN_HISTORY usable ratings (or unknown users with no history) get
    popularity-based scores (cold-start fallback). Only movies with at least 50 ratings in the full MovieLens
    data are candidates; fewer than top_k movies can be returned if KNN has no neighbour evidence for more.
    """
    if method not in ("knn", "svd"):
        raise ValueError("method must be 'knn' or 'svd'")
    model = model or get_model()
    idx, r, u = _history_vectors(model, user_id, rating_history)

    if len(idx) < MIN_HISTORY:
        score = model.log_pop.astype(np.float64).copy()
    elif method == "knn":
        score = _knn_scores(model, idx, r)
    else:
        score = _svd_scores(model, idx, r, u)
    score[idx] = -np.inf                                         # never return already-rated movies

    k = min(top_k, int(np.isfinite(score).sum()))
    if k == 0:
        return {}
    top = np.argpartition(-score, k - 1)[:k]
    top = top[np.argsort(-score[top])]
    vals = score[top]
    if normalize:
        lo, hi = vals.min(), vals.max()
        vals = (vals - lo) / (hi - lo) if hi > lo else np.ones_like(vals)
    return {int(model.movie_ids[i]): float(v) for i, v in zip(top, vals)}


def recommend_knn(user_id, n=10, model=None):
    """Top-n unseen movies by Item-KNN. Returns [(movieId, raw_score), ...]; unknown user -> popularity list."""
    return list(get_cf_scores(user_id, "knn", n, normalize=False, model=model).items())


def recommend_svd(user_id, n=10, model=None):
    """Top-n unseen movies by SVD. Returns [(movieId, raw_score), ...]; unknown user -> popularity list."""
    return list(get_cf_scores(user_id, "svd", n, normalize=False, model=model).items())


# ----------------------------------------------------------------------------- score blocks (used by evaluation)
def score_block_knn(model):
    NT = model.N.T.tocsr()
    liked = model.R.copy(); liked.data = (liked.data >= LIKE_THRESHOLD).astype(np.float32); liked.eliminate_zeros()
    rated = model.R.copy(); rated.data[:] = 1
    def fn(ub):
        s = (liked[ub] @ NT).toarray(); d = (rated[ub] @ NT).toarray()
        s[d <= 0] = -np.inf
        return s
    return fn


def score_block_svd(model):
    return lambda ub: model.P[ub] @ model.Q.T + model.bi + POP_WEIGHT * model.log_pop


def score_block_popularity(model):
    return lambda ub: np.tile(model.log_pop.astype(np.float32), (len(ub), 1))


# ----------------------------------------------------------------------------- demo / self-check
if __name__ == "__main__":
    import sys
    sys.path.insert(0, REPO_ROOT)
    from evaluation.collaborative_metrics import build_relevance_matrix, ranking_metrics, rmse_mae, summarise

    m = get_model()
    print("\n=== Sample recommendations (Top-5) ===")
    for uid in [m.user_ids[0], m.user_ids[5000], m.user_ids[20000]]:
        liked = m.train[(m.train.userId == uid) & (m.train.rating >= 4.0)].movieId.head(3)
        print(f"\nUser {uid} (liked e.g.: {', '.join(m.titles[liked])})")
        for method in ("knn", "svd"):
            recs = get_cf_scores(uid, method, 5)
            print(f"  {method.upper()}: " + " | ".join(f"{m.titles[mid]} ({sc:.2f})" for mid, sc in recs.items()))

    print("\n=== Interface checks ===")
    uid = m.user_ids[0]; rated = set(m.train[m.train.userId == uid].movieId)
    for method in ("knn", "svd"):
        out = get_cf_scores(uid, method, 100)
        assert len(out) == 100 and min(out.values()) == 0.0 and max(out.values()) == 1.0
        assert not (set(out) & rated), "rated movie returned"
    assert len(get_cf_scores(None, "svd", 5, rating_history={1: 5.0, 2: 4.0})) == 5      # cold start
    assert len(get_cf_scores(123456789, "knn", 5)) == 5                                   # unknown user
    try:
        get_cf_scores(uid, "bad"); raise SystemExit("invalid method was accepted")
    except ValueError:
        pass
    print("OK: rated movies excluded, scores in [0,1], cold start and invalid input handled")

    print("\n=== Evaluation on the held-out test set (should match the notebook) ===")
    rel = build_relevance_matrix(m.te_u, m.te_i, m.te_r, m.R.shape, RELEVANT_THRESHOLD)
    for name, fn in [("Item-KNN (K=50)", score_block_knn(m)), ("SVD (50 factors)", score_block_svd(m)),
                     ("Popularity", score_block_popularity(m))]:
        _, pu = ranking_metrics(fn, m.R, rel)
        s = summarise(pu)
        print(f"{name:18s} P@10 {s['Precision@10']:.4f} | R@10 {s['Recall@10']:.4f} | NDCG@10 {s['NDCG@10']:.4f}")
    tp = m.mu + m.bu[m.te_u] + m.bi[m.te_i] + _dot_rows(m.P, m.Q, m.te_u, m.te_i)
    rm, ma = rmse_mae(tp, m.te_r)
    print(f"SVD rating error   RMSE {rm:.4f} | MAE {ma:.4f}")
    print("Notebook reference: KNN 0.2150/0.1843/0.2880 | SVD 0.1440/0.1130/0.1844 | Pop 0.1192/0.0990/0.1539 | SVD RMSE 0.7959 MAE 0.6011")