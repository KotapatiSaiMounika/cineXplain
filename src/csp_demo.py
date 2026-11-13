import pandas as pd
from src.constraints import MovieConstraintChecker

movies_data = pd.DataFrame([
    {'movieId': 1, 'title': 'The Conjuring', 'genres': 'Horror|Thriller', 'year': 2013, 'franchise_id': 10},
    {'movieId': 2, 'title': 'Inception', 'genres': 'Sci-Fi|Action', 'year': 2010, 'franchise_id': None},
    {'movieId': 3, 'title': 'Toy Story 2', 'genres': 'Animation|Children', 'year': 1999, 'franchise_id': 20},
    {'movieId': 4, 'title': 'Toy Story 3', 'genres': 'Animation|Children', 'year': 2010, 'franchise_id': 20}
])

mmr_output = {1: 0.95, 2: 0.90, 3: 0.85, 4: 0.82}

prefs = {
    'excluded_genres': ['Horror'],
    'min_year': 2000,
    'max_sequels': 1
}

checker = MovieConstraintChecker(movies_data)
filtered_results = checker.apply_constraints(mmr_output, prefs)
print("Final Filtered Recommendations:", filtered_results)
