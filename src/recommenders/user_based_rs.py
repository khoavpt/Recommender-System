import numpy as np

from utils.similarity_measure import pure_cosine_similarity, pearson_correlation
from recommenders.memory_based_recommender_system import MemoryBasedRecommenderSystem
    
class UserBasedRecommenderSystem(MemoryBasedRecommenderSystem):
    def __init__(self, ratings_df, model_size,similarity_measure='adjusted_cosine'):
        super().__init__(ratings_df, model_size, similarity_measure)
    
    def fit(self):
        """
        Calculate similarity matrix between users and store it in self.similarity_matrix
        """
        users_count = len(self.unique_users)
        self.similarity_matrix = np.zeros((users_count, users_count))

        for user_i in self.unique_users:
            user_i_index = self.user_to_index[user_i]
            for user_j in self.unique_users:
                user_j_index = self.user_to_index[user_j]
                if user_i > user_j:
                    self.similarity_matrix[user_i_index, user_j_index] = self.similarity_matrix[user_j_index, user_i_index]
                    continue
                elif user_i == user_j:
                    continue
                else:
                    movies_rated_by_both = np.nonzero((~np.isnan(self.rating_matrix[user_i_index, :])) & (~np.isnan(self.rating_matrix[user_j_index, :])))[0]
                    user_i_vec = self.rating_matrix[user_i_index, movies_rated_by_both]
                    user_j_vec = self.rating_matrix[user_j_index, movies_rated_by_both]

                    similarity = 0
                    if self.similarity_measure == 'pearson_correlation':
                        similarity = pearson_correlation(user_i_vec, user_j_vec)
                    elif self.similarity_measure == 'pure_cosine':
                        similarity = pure_cosine_similarity(user_i_vec, user_j_vec)
                    elif self.similarity_measure == 'adjusted_cosine':
                        m = np.nanmean(self.rating_matrix[:, movies_rated_by_both], axis=0)
                        user_i_vec = user_i_vec - m
                        user_j_vec = user_j_vec - m
                        similarity = pure_cosine_similarity(user_i_vec, user_j_vec)

                    self.similarity_matrix[user_i_index, user_j_index] = similarity
    
    def predict_rating(self, user_id, movie_id, neighborhood_size=30):
        """
        Predict rating of user_id for movie_id
        """
        user_index = self.user_to_index.get(user_id)
        movie_index = self.movie_to_index.get(movie_id)

        if user_index is None and movie_index is None:
            return np.nanmean(self.rating_matrix)
        elif movie_index is None:
            return np.nanmean(self.rating_matrix[user_index, :])
        elif user_index is None:
            return np.nanmean(self.rating_matrix[:, movie_index])
        elif not np.isnan(self.rating_matrix[user_index, movie_index]):
            return self.rating_matrix[user_index, movie_index]

        movie_ratings = self.rating_matrix[:, movie_index]
        users_rated_movie_index = np.nonzero(~np.isnan(movie_ratings))[0]
        users_similarities = self.similarity_matrix[user_index, users_rated_movie_index]

        top_k_similar_indexes = np.argsort(users_similarities)[-neighborhood_size:]
        top_k_similarities = users_similarities[top_k_similar_indexes]
        top_k_movie_ratings = movie_ratings[users_rated_movie_index[top_k_similar_indexes]]

        denominator = np.linalg.norm(top_k_similarities, ord=1)
        rating_prediction = np.dot(top_k_similarities, top_k_movie_ratings) / denominator if denominator != 0 else np.nanmean(self.rating_matrix[user_index, :])

        return rating_prediction