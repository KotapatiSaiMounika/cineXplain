import pandas as pd

class MovieConstraintChecker:
    def __init__(self, movies_df: pd.DataFrame):

        self.movies_df = movies_df.set_index('movieId')

    def apply_constraints(self, ranked_movies: dict, user_preferences: dict) -> dict:

        filtered_movies = {}
        franchise_counts = {}

        excluded_genres = [g.lower() for g in user_preferences.get('excluded_genres', [])]
        min_year = user_preferences.get('min_year', 1900)
        max_year = user_preferences.get('max_year', 2030)
        max_sequels = user_preferences.get('max_sequels', float('inf'))

        for movie_id, score in ranked_movies.items():
            if movie_id not in self.movies_df.index:
                continue

            movie = self.movies_df.loc[movie_id]
            movie_genres = [g.lower() for g in str(movie.get('genres', '')).split('|')]
            movie_year = int(movie.get('year', 1900))
            franchise_id = movie.get('franchise_id', None)

            if any(exc in movie_genres for exc in excluded_genres):
                continue  # Violation: skip

            if not (min_year <= movie_year <= max_year):
                continue  # Violation: skip

            if franchise_id is not None:
                current_count = franchise_counts.get(franchise_id, 0)
                if current_count >= max_sequels:
                    continue  # Violation: skip
                franchise_counts[franchise_id] = current_count + 1

            filtered_movies[movie_id] = score

        if not filtered_movies and ranked_movies:
            print("[Warning] Constraints were too strict; returning top recommendations with relaxed rules.")
            return dict(list(ranked_movies.items())[:3])

        return filtered_movies
