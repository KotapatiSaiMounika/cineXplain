import pandas as pd


class MovieConstraintChecker:
    def __init__(self, movies_df: pd.DataFrame):
        self.movies_df = movies_df.set_index('movieId')

    @staticmethod
    def _genres(movie) -> list:
        return [g.strip().lower() for g in str(movie.get('genres', '')).split('|')]

    @staticmethod
    def _franchise(movie):
        """Franchise name from the 'collection' column (older data used 'franchise_id').
        Blank or missing values mean 'no franchise'."""
        value = movie.get('collection', movie.get('franchise_id', None))
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        value = str(value).strip()
        return value or None

    def apply_constraints(self, ranked_movies: dict, user_preferences: dict) -> dict:

        filtered_movies = {}
        franchise_counts = {}

        excluded_genres = [g.lower() for g in (user_preferences.get('excluded_genres') or [])]
        min_year = user_preferences.get('min_year') or 1900
        max_year = user_preferences.get('max_year') or 2030
        max_sequels = user_preferences.get('max_sequels') or float('inf')

        for movie_id, score in ranked_movies.items():
            if movie_id not in self.movies_df.index:
                continue

            movie = self.movies_df.loc[movie_id]
            movie_genres = self._genres(movie)
            movie_year = int(movie.get('year', 1900))
            franchise = self._franchise(movie)

            if any(exc in movie_genres for exc in excluded_genres):
                continue  # Violation: skip

            if not (min_year <= movie_year <= max_year):
                continue  # Violation: skip

            if franchise is not None:
                current_count = franchise_counts.get(franchise, 0)
                if current_count >= max_sequels:
                    continue  # Violation: skip
                franchise_counts[franchise] = current_count + 1

            filtered_movies[movie_id] = score

        if not filtered_movies and ranked_movies:
            print("[Warning] Constraints were too strict; relaxing year and franchise rules "
                  "(excluded genres are never relaxed).")
            relaxed = {}
            for movie_id, score in ranked_movies.items():
                if movie_id not in self.movies_df.index:
                    continue
                if any(exc in self._genres(self.movies_df.loc[movie_id]) for exc in excluded_genres):
                    continue
                relaxed[movie_id] = score
                if len(relaxed) == 3:
                    break
            return relaxed

        return filtered_movies
