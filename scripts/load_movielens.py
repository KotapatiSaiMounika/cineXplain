
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/movielens")

# Load datasets
movies = pd.read_csv(DATA_DIR / "movie.csv")
ratings = pd.read_csv(DATA_DIR / "rating.csv")
tags = pd.read_csv(DATA_DIR / "tag.csv")
links = pd.read_csv(DATA_DIR / "link.csv")
genome_tags = pd.read_csv(DATA_DIR / "genome_tags.csv")
genome_scores = pd.read_csv(DATA_DIR / "genome_scores.csv")

# Display dataset information
print("CINEXPLAIN - MOVIELENS DATASET")
print("=" * 40)

print("Movies:", movies.shape)
print("Ratings:", ratings.shape)
print("Tags:", tags.shape)
print("Links:", links.shape)
print("Genome Tags:", genome_tags.shape)
print("Genome Scores:", genome_scores.shape)

print("\nTotal users:", ratings["userId"].nunique())
print("Total rated movies:", ratings["movieId"].nunique())

print("\nMovie sample:")
print(movies.head())

print("\nRating sample:")
print(ratings.head())
