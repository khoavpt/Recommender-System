import numpy as np

from recommenders.recommender_system import RecommenderSystem
    
class NonPersonalizedRecommenderSystem(RecommenderSystem):
    def __init__(self, ratings_df):
        super().__init__(ratings_df)
        self.average_rating_for_each_movie = {}

    def fit(self):
        self.average_rating_for_each_movie = np.nanmean(self.rating_matrix, axis=0)

    def predict_rating(self, user_id, movie_id, neighborhood_size=30):
        """
        Predict rating of user_id for movie_id
        """
        movie_index = self.movie_to_index.get(movie_id)

        if movie_index is None:
            return np.nanmean(self.average_rating_for_each_movie)

        return self.average_rating_for_each_movie[movie_index]
