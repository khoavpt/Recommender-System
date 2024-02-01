import numpy as np

from utils.similarity_measure import cosine_similarity, pearson_correlation
from recommenders.recommender_system import RecommenderSystem

class ItemBasedRegressionRecommenderSystem(RecommenderSystem):
    def __init__(self, ratings_df, k_neighbors, similarity_measure=cosine_similarity):
        super().__init__(ratings_df, k_neighbors, similarity_measure)
        self.top_k_similar_movie_indexes = {movie_id: None for movie_id in self.unique_movies}
        self.weights_matrix = None

    def fit(self):
        """
        Calculate similarity matrix between items and store it in self.similarity_matrix
        """
        self.weights_matrix = np.zeros((len(self.unique_movies), self.k_neighbors))
        for movie_i in self.unique_movies:
            top_k_similarities_for_movie_i = []
            for movie_j in self.unique_movies:
                if movie_i == movie_j:
                    continue
                else:
                    users_rated_movie_i_index = np.nonzero(self.rating_matrix[:, self.movie_to_index[movie_i]] != 0)[0]
                    users_rated_both = np.nonzero((self.rating_matrix[:, self.movie_to_index[movie_i]] != 0) & (self.rating_matrix[:, self.movie_to_index[movie_j]] != 0))[0]
                    movie_i_vec = self.rating_matrix[users_rated_both, self.movie_to_index[movie_i]]
                    movie_j_vec = self.rating_matrix[users_rated_both, self.movie_to_index[movie_j]]
                    similarity = self.similarity_measure(movie_i_vec, movie_j_vec)

                    top_k_similarities_for_movie_i.append((self.movie_to_index[movie_j], similarity))
            # print(np.array(sorted(top_k_similarities_for_movie_i, key=lambda x: x[1], reverse=True)[:self.k_neighbors], dtype=int)[:, 0])
            self.top_k_similar_movie_indexes[movie_i] = np.array(sorted(top_k_similarities_for_movie_i, key=lambda x: x[1], reverse=True)[:self.k_neighbors], dtype=int)[:, 0]
            X = self.rating_matrix[users_rated_movie_i_index, :][:, self.top_k_similar_movie_indexes[movie_i]]
            y = self.rating_matrix[users_rated_movie_i_index, self.movie_to_index[movie_i]]
            self.weights_matrix[self.movie_to_index[movie_i], :] = np.linalg.pinv(X.T @ X) @ X.T @ y
    
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

        user_ratings_on_similar_movies = self.rating_matrix[user_index, self.top_k_similar_movie_indexes[movie_id]]
        rating_prediction = self.weights_matrix[movie_index, :] @ user_ratings_on_similar_movies
        return rating_prediction