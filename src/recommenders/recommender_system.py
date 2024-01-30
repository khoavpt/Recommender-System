import numpy as np
import pickle

from sklearn.metrics import mean_absolute_error

class RecommenderSystem:
    def __init__(self, ratings_df):
        self.unique_movies = ratings_df['movieId'].unique()
        self.movie_to_index = {movie_id: index for index, movie_id in enumerate(self.unique_movies)}
        self.unique_users = ratings_df['userId'].unique()
        self.user_to_index = {user_id: index for index, user_id in enumerate(self.unique_users)}
        
        # Create a rating matrix with rows as users and columns as movies
        self.rating_matrix = np.zeros((len(self.unique_users), len(self.unique_movies)))
        for _, row in ratings_df.iterrows():
            self.rating_matrix[self.user_to_index[row['userId']], self.movie_to_index[row['movieId']]] = row['rating']
    
    # Abstract method
    def fit(self):
        """
        Calculate similarity matrix and store it in self.similarity_matrix
        """
        raise NotImplementedError

    # Abstract method
    def predict_rating(self, user_id, movie_id, k_neighbors=10):
        """
        Predict rating of user_id for movie_id
        """
        raise NotImplementedError
    
    def recommend_items(self, user_id, top_n, k_neighbors=10):
        """
        Recommend top_n items for user_id
        """
        user_index = self.user_to_index.get(user_id)
        if user_index is None:
            return []
        user_ratings = self.rating_matrix[user_index, :]
        movies_not_rated_by_user_index = np.nonzero(user_ratings == 0)[0]
        movie_ratings = []
        for movie_index in movies_not_rated_by_user_index:
            rating = self.predict_rating(user_id, self.unique_movies[movie_index], k_neighbors)
            movie_ratings.append((movie_index, rating))
        return [self.unique_movies[movie_index] for movie_index, _ in sorted(movie_ratings, key=lambda x: x[1], reverse=True)[:top_n]]
    
    def evaluate(self, test_df, k_neighbors=10, metric=mean_absolute_error):
        y_pred =[]
        y_true = test_df['rating'].values
        for _, row in test_df.iterrows():
            y_pred.append(self.predict_rating(row['userId'], row['movieId'], k_neighbors))
        return metric(y_true, y_pred)
            
    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod   
    def load_model(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
        
