import numpy as np

from utils.similarity_measure import cosine_similarity, pearson_correlation
from recommenders.recommender_system import RecommenderSystem
    
class UserBasedRecommenderSystem(RecommenderSystem):
    def __init__(self, ratings_df, k_neighbors,similarity_measure=cosine_similarity):
        super().__init__(ratings_df, k_neighbors, similarity_measure)
        self.similarity_matrix = None
    
    def fit(self):
        """
        Calculate similarity matrix between users and store it in self.similarity_matrix
        """
        users_count = len(self.unique_users)
        self.similarity_matrix = np.zeros((users_count, users_count))

        for user_i in self.unique_users:
            for user_j in self.unique_users:
                if user_i > user_j:
                    self.similarity_matrix[self.user_to_index[user_i], self.user_to_index[user_j]] = self.similarity_matrix[self.user_to_index[user_j], self.user_to_index[user_i]]
                    continue
                elif user_i == user_j:
                    self.similarity_matrix[self.user_to_index[user_i], self.user_to_index[user_j]] = 1
                    continue
                else:
                    movies_rated_by_both = np.nonzero((self.rating_matrix[self.user_to_index[user_i], :] != 0) & (self.rating_matrix[self.user_to_index[user_j], :] != 0))[0]
                    user_i_vec = self.rating_matrix[self.user_to_index[user_i], movies_rated_by_both]
                    user_j_vec = self.rating_matrix[self.user_to_index[user_j], movies_rated_by_both]
                    similarity = self.similarity_measure(user_i_vec, user_j_vec)

                    self.similarity_matrix[self.user_to_index[user_i], self.user_to_index[user_j]] = similarity
    
    def predict_rating(self, user_id, movie_id):
        """
        Predict rating of user_id for movie_id
        """
        user_index = self.user_to_index.get(user_id)
        movie_index = self.movie_to_index.get(movie_id)

        if user_index is None and movie_index is None:
            return np.mean(self.rating_matrix)
        elif movie_index is None:
            return np.mean(self.rating_matrix[user_index, :])
        elif user_index is None:
            return np.mean(self.rating_matrix[:, movie_index])
        elif self.rating_matrix[user_index, movie_index] != 0:
            return self.rating_matrix[user_index, movie_index]

        movie_ratings = self.rating_matrix[:, movie_index]
        users_rated_movie_index = np.nonzero(movie_ratings != 0)[0]
        users_similarities = self.similarity_matrix[user_index, users_rated_movie_index]

        top_k_similar_indexes = np.argsort(users_similarities)[-self.k_neighbors:]
        top_k_similarities = users_similarities[top_k_similar_indexes]
        top_k_movie_ratings = movie_ratings[users_rated_movie_index[top_k_similar_indexes]]

        denominator = np.sum(top_k_similarities)
        rating_prediction = np.dot(top_k_similarities, top_k_movie_ratings) / denominator if denominator != 0 else np.mean(self.rating_matrix[user_index, :])

        return rating_prediction