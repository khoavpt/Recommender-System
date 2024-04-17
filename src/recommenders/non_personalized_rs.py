import numpy as np
import torch

from recommenders.recommender_system import RecommenderSystem
    
class NonPersonalizedRecommenderSystem(RecommenderSystem):
    def __init__(self, ratings_df, precompute_rating_matrix_path=None):
        super().__init__(ratings_df, precompute_rating_matrix_path)
        self.average_rating_for_each_movie = {}

    def fit(self):
        self.average_rating_for_each_movie = torch.nanmean(self.rating_matrix, axis=0)

    def predict_rating(self, user_id, movie_id):
        """
        Predict rating of user_id for movie_id
        """
        movie_index = self.movie_to_index.get(movie_id)

        if movie_index is None:
            return torch.nanmean(self.average_rating_for_each_movie)

        return self.average_rating_for_each_movie[movie_index]
