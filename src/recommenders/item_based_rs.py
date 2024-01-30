import numpy as np

from utils.similarity_measure import cosine_similarity, pearson_correlation
from recommenders.recommender_system import RecommenderSystem

class ItemBasedRecommenderSystem(RecommenderSystem):
    def __init__(self, ratings_df, similarity_measure=cosine_similarity):
        super().__init__(ratings_df)
        self.similarity_measure = similarity_measure
        self.similarity_matrix = None
    
    def fit(self):
        """
        Calculate similarity matrix between items and store it in self.similarity_matrix
        """
        movies_count = len(self.unique_movies)
        self.similarity_matrix = np.zeros((movies_count, movies_count))

        for movie_i in self.unique_movies:
            for movie_j in self.unique_movies:
                if movie_i > movie_j:
                    self.similarity_matrix[self.movie_to_index[movie_i], self.movie_to_index[movie_j]] = self.similarity_matrix[self.movie_to_index[movie_j], self.movie_to_index[movie_i]]
                    continue
                elif movie_i == movie_j:
                    self.similarity_matrix[self.movie_to_index[movie_i], self.movie_to_index[movie_j]] = 1
                    continue
                else:
                    users_rated_both = np.nonzero((self.rating_matrix[:, self.movie_to_index[movie_i]] != 0) & (self.rating_matrix[:, self.movie_to_index[movie_j]] != 0))[0]
                    movie_i_vec = self.rating_matrix[users_rated_both, self.movie_to_index[movie_i]]
                    movie_j_vec = self.rating_matrix[users_rated_both, self.movie_to_index[movie_j]]
                    similarity = self.similarity_measure(movie_i_vec, movie_j_vec)

                    self.similarity_matrix[self.movie_to_index[movie_i], self.movie_to_index[movie_j]] = similarity
                    
    def predict_rating(self, user_id, movie_id, k_neighbors=10):
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

        user_ratings = self.rating_matrix[user_index, :]
        movies_rated_by_user_index = np.nonzero(user_ratings != 0)[0]
        movie_similarities = self.similarity_matrix[movie_index, movies_rated_by_user_index]

        top_k_similar_indexes = np.argsort(movie_similarities)[-k_neighbors:]
        top_k_similarities = movie_similarities[top_k_similar_indexes]
        top_k_user_ratings = user_ratings[movies_rated_by_user_index[top_k_similar_indexes]]

        denominator = np.sum(top_k_similarities)
        rating_prediction = np.dot(top_k_similarities, top_k_user_ratings) / denominator if denominator != 0 else np.mean(self.rating_matrix[:, movie_index])

        return rating_prediction
