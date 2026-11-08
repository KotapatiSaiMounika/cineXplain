import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from src.collaborative_recommender import (
    prepare_interactions,
    build_item_similarity,
    neighbour_matrix,
    K_NEIGHBOURS,
)

BASE = Path("data/movielens")
CACHE = Path("data/model_outputs/temporal_knn")
CACHE.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(BASE / "splits/train_ratings.csv")
test = pd.read_csv(BASE / "splits/test_ratings.csv")
movies = pd.read_csv(BASE / "movie.csv")

# Keep movies with at least 50 ratings in temporal training data
counts = train.groupby("movieId").size()
keep = counts[counts >= 50].index
train = train[train.movieId.isin(keep)].copy()

# Keep test rows only for movies available in training
test = test[test.movieId.isin(keep)].copy()

print("Preparing temporal interactions...")
data = prepare_interactions(train, test, movies)

idx_path = CACHE / "neighbours_idx.npy"
sim_path = CACHE / "neighbours_sim.npy"

if idx_path.exists() and sim_path.exists():
    print("Loading saved temporal neighbours...")
    idx = np.load(idx_path)
    sim = np.load(sim_path)
else:
    print("Building temporal Item-KNN neighbours...")
    idx, sim = build_item_similarity(
        data.R, data.user_mean, K_NEIGHBOURS
    )
    np.save(idx_path, idx)
    np.save(sim_path, sim)

data.nbr_idx = idx
data.nbr_sim = sim
data.N = neighbour_matrix(idx, sim, K_NEIGHBOURS)

import pickle
with open(CACHE / "temporal_knn_model.pkl", "wb") as f:
    pickle.dump(data, f)

print("Temporal KNN model saved.")
print("Users:", data.n_users)
print("Movies:", data.n_movies)
print("Training ratings:", data.R.nnz)
