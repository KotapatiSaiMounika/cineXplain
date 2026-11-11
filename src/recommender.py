from src.content_recommender import get_content_scores
from src.collaborative_recommender import get_cf_scores
from src.hybrid_recommender import compute_hybrid_scores
from src.mmr import apply_mmr
from src.constraints import MovieConstraintChecker

class CineXplainSystem:
    def __init__(self, movies_df, ratings_df):
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.constraint_checker = MovieConstraintChecker(movies_df)

    def recommend(self, user_id, user_preferences, top_k=10):

        content_scores = get_content_scores(user_id, self.movies_df)

        cf_scores = get_cf_scores(user_id, self.ratings_df)

        hybrid_scores = compute_hybrid_scores(user_id, content_scores, cf_scores, self.ratings_df)

        diverse_ranked = apply_mmr(hybrid_scores, self.movies_df, lambda_param=0.7, top_n=50)

        final_filtered = self.constraint_checker.apply_constraints(diverse_ranked, user_preferences)

        final_top_k = dict(list(final_filtered.items())[:top_k])

        return final_top_k
