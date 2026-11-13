import pandas as pd

from src.constraints import MovieConstraintChecker


def make_checker():
    movies = pd.DataFrame([
        {"movieId": 1, "title": "The Conjuring", "genres": "Horror|Thriller", "year": 2013, "collection": "The Conjuring Collection"},
        {"movieId": 2, "title": "Inception", "genres": "Action|Sci-Fi", "year": 2010, "collection": ""},
        {"movieId": 3, "title": "Toy Story 2", "genres": "Animation|Children", "year": 1999, "collection": "Toy Story Collection"},
        {"movieId": 4, "title": "Toy Story 3", "genres": "Animation|Children", "year": 2010, "collection": "Toy Story Collection"},
    ])
    return MovieConstraintChecker(movies)


RANKED = {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6}


def test_genre_exclusion():
    result = make_checker().apply_constraints(RANKED, {"excluded_genres": ["Horror"]})
    assert 1 not in result
    assert 2 in result


def test_year_range():
    result = make_checker().apply_constraints(RANKED, {"min_year": 2005, "max_year": 2015})
    assert set(result) == {1, 2, 4}


def test_franchise_cap():
    result = make_checker().apply_constraints(RANKED, {"max_sequels": 1})
    toy_story = [m for m in result if m in (3, 4)]
    assert len(toy_story) == 1


def test_none_values_do_not_crash():
    result = make_checker().apply_constraints(
        RANKED, {"excluded_genres": None, "min_year": None, "max_year": None, "max_sequels": None}
    )
    assert set(result) == {1, 2, 3, 4}


def test_blank_collection_not_treated_as_franchise():
    movies = pd.DataFrame([
        {"movieId": 1, "title": "A", "genres": "Drama", "year": 2000, "collection": ""},
        {"movieId": 2, "title": "B", "genres": "Drama", "year": 2001, "collection": ""},
        {"movieId": 3, "title": "C", "genres": "Drama", "year": 2002, "collection": ""},
    ])
    result = MovieConstraintChecker(movies).apply_constraints(
        {1: 0.9, 2: 0.8, 3: 0.7}, {"max_sequels": 1}
    )
    assert set(result) == {1, 2, 3}


def test_fallback_never_returns_excluded_genre():
    # Year range matches nothing, so the fallback kicks in, but Horror must stay excluded.
    result = make_checker().apply_constraints(
        RANKED, {"excluded_genres": ["Horror"], "min_year": 1900, "max_year": 1950}
    )
    assert result
    assert 1 not in result
    