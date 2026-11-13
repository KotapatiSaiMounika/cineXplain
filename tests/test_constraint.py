# tests/test_constraints.py
import pytest
import pandas as pd
from src.constraints import MovieConstraintChecker

@pytest.fixture
func sample_movies_df():
    return pd.DataFrame([
        {'movieId': 1, 'title': 'The Conjuring', 'genres': ['Horror', 'Thriller'], 'year': 2013, 'franchise_id': 'conjuring'},
        {'movieId': 2, 'title': 'Inception', 'genres': ['Action', 'Sci-Fi'], 'year': 2010, 'franchise_id': None},
        {'movieId': 3, 'title': 'Toy Story 2', 'genres': ['Animation', 'Children'], 'year': 1999, 'franchise_id': 'toystory'},
        {'movieId': 4, 'title': 'Toy Story 3', 'genres': ['Animation', 'Children'], 'year': 2010, 'franchise_id': 'toystory'}
    ])

def test_genre_exclusion(sample_movies_df):
    constraints = {'excluded_genres': ['Horror']}
    checker = MovieConstraintChecker(constraints)
    
    candidates = sample_movies_df.to_dict('records')
    filtered = checker.filter_candidates(candidates, top_k=10)
    
    # Ensure 'The Conjuring' (Horror) is excluded
    movie_titles = [m['title'] for m in filtered]
    assert 'The Conjuring' not in movie_titles
    assert 'Inception' in movie_titles

def test_franchise_sequel_cap(sample_movies_df):
    constraints = {'max_per_franchise': 1}
    checker = MovieConstraintChecker(constraints)
    
    candidates = sample_movies_df.to_dict('records')
    filtered = checker.filter_candidates(candidates, top_k=10)
    
    # Should only allow 1 Toy Story movie due to franchise cap
    toystory_count = sum(1 for m in filtered if m.get('franchise_id') == 'toystory')
    assert toystory_count == 1
