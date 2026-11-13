# evaluation/final_comparison.py

def evaluate_constraint_satisfaction(recommendations_dict, movies_df, user_preferences):
    """
    Calculates what percentage of final recommendations successfully 
    complied with the user's constraints.
    """
    total = len(recommendations_dict)
    if total == 0:
        return 0.0
        
    passed = 0
    excluded_genres = [g.lower() for g in user_preferences.get('excluded_genres', [])]
    min_year = user_preferences.get('min_year', 1900)
    max_year = user_preferences.get('max_year', 2030)

    for movie_id in recommendations_dict.keys():
        if movie_id not in movies_df.index:
            continue
        movie = movies_df.loc[movie_id]
        genres = [g.lower() for g in str(movie.get('genres', '')).split('|')]
        year = int(movie.get('year', 1900))

        has_excluded = any(exc in genres for exc in excluded_genres)
        in_year_range = min_year <= year <= max_year

        if not has_excluded and in_year_range:
            passed += 1

    satisfaction_rate = (passed / total) * 100.0
    return satisfaction_rate
