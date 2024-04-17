import numpy as np
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import pickle

from utils.similarity_measure import pure_cosine_similarity, pearson_correlation
from recommenders.memory_based_rs import MemoryBasedRecommenderSystem

class Data(torch.utils.data.Dataset):
    def __init__(self, rating_matrix):
        self.rating_matrix = rating_matrix
        self.users_count = rating_matrix.shape[0]
        self.items_count = rating_matrix.shape[1]
        self.valid_pairs = [(u, i) for u in range(self.users_count) for i in range(self.items_count) if not torch.isnan(self.rating_matrix[u, i])]

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        """
        Get user and item index of rated
        """
        return self.valid_pairs[idx]

class ItemRegressionModel(nn.Module):
    def __init__(self, rating_matrix, qtus, neighborhood_size):
        super().__init__()
        self.users_count = rating_matrix.shape[0]
        self.items_count = rating_matrix.shape[1]
        self.rating_matrix = rating_matrix
        self.neighborhood_size = neighborhood_size
        self.weight = nn.Parameter(torch.randn(self.items_count, self.items_count))
        self.b_user = nn.Parameter(torch.randn(self.users_count))
        self.b_item = nn.Parameter(torch.randn(self.items_count))
        self.qtus = qtus
        
        nn.init.kaiming_normal_(self.weight)
        nn.init.normal_(self.b_user)
        nn.init.normal_(self.b_item)

    def forward(self, x):
        batch_rut_hat = []
        for u, t in zip(x[0].tolist(), x[1].tolist()):
            qtu = self.qtus[(u, t)]
            w_jt = self.weight[qtu, t]
            r_uj = self.rating_matrix[u, qtu]
            rut_hat = self.b_user[u] + self.b_item[t] + torch.dot(w_jt, r_uj - self.b_user[u] - self.b_item[qtu]) / self.neighborhood_size
            batch_rut_hat.append(rut_hat)
        return torch.stack(batch_rut_hat)

        # batch_qtu = torch.stack([nn.functional.pad(torch.tensor(self.qtus[(u, t)]), (0, self.neighborhood_size - len(self.qtus[(u, t)]))) for u, t in zip(x[0].tolist(), x[1].tolist())])
        # batch_wjt = self.weight[x[1].unsqueeze(dim=1), batch_qtu]
        # batch_ruj = self.rating_matrix[x[0].unsqueeze(dim=1), batch_qtu]

        # batch_rut_hat = self.b_user[x[0]] + self.b_item[x[1]] + torch.sum(batch_wjt * (batch_ruj - self.b_user[x[0]].unsqueeze(dim=1) - self.b_item[batch_qtu]), dim=1) / self.neighborhood_size
        # return batch_rut_hat

class ItemBasedRegressionRecommenderSystem(MemoryBasedRecommenderSystem):
    def __init__(self, ratings_df, neighborhood_size, precompute_qtus_path, precompute_rating_matrix_path=None, precompute_similarity_matrix_path=None, similarity_measure='adjusted_cosine', optimizer=torch.optim.Adam, epochs=20, lr=0.0005, weight_decay=0):
        super().__init__(ratings_df, neighborhood_size, precompute_rating_matrix_path, precompute_similarity_matrix_path, similarity_measure)

        with open(precompute_qtus_path, 'rb') as f:
            self.qtus = pickle.load(f)
        
        self.qtus = {key: value[:neighborhood_size] for key, value in self.qtus.items() if len(value) > 0}
        self.epochs = epochs
        self.model = ItemRegressionModel(self.rating_matrix, self.qtus, neighborhood_size)
        self.optimizer = optimizer(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def calculate_similarity_matrix(self):
        """
        Calculate similarity matrix between items and store it in self.similarity_matrix
        """
        movies_count = len(self.unique_movies)
        self.similarity_matrix = np.zeros((movies_count, movies_count))

        for movie_i in self.unique_movies:
            movie_i_index = self.movie_to_index[movie_i]
            for movie_j in self.unique_movies:
                movie_j_index = self.movie_to_index[movie_j]
                if movie_i > movie_j:
                    self.similarity_matrix[movie_i_index, movie_j_index] = self.similarity_matrix[movie_j_index, movie_i_index]
                    continue
                elif movie_i == movie_j:
                    continue
                else:
                    users_rated_both = np.nonzero(~np.isnan(self.rating_matrix[:, movie_i_index]) & ~np.isnan(self.rating_matrix[:, movie_j_index]))[0]
                    movie_i_vec = self.rating_matrix[users_rated_both, movie_i_index]
                    movie_j_vec = self.rating_matrix[users_rated_both, movie_j_index]

                    similarity = 0
                    if self.similarity_measure == 'pearson_correlation':
                        similarity = pearson_correlation(movie_i_vec, movie_j_vec)
                    elif self.similarity_measure == 'pure_cosine':
                        similarity = pure_cosine_similarity(movie_i_vec, movie_j_vec)
                    elif self.similarity_measure == 'adjusted_cosine':
                        movie_i_vec = movie_i_vec - np.nanmean(self.rating_matrix[users_rated_both, :], axis=1)
                        movie_j_vec = movie_j_vec - np.nanmean(self.rating_matrix[users_rated_both, :], axis=1)
                        similarity = pure_cosine_similarity(movie_i_vec, movie_j_vec)

                    self.similarity_matrix[movie_i_index, movie_j_index] = similarity

        self.similarity_matrix = torch.from_numpy(self.similarity_matrix).float()

    def train_model(self):
        """
        Train the model
        """
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.rating_matrix = self.rating_matrix.to(device)
        self.model.rating_matrix = self.model.rating_matrix.to(device)

        data = Data(self.rating_matrix)
        train_loader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True)

        criterion = nn.MSELoss()
        print('start training!')
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for u, i in train_loader:
                u = u.to(device)
                i = i.to(device)
                target = self.rating_matrix[u, i]
                self.optimizer.zero_grad()
                output = self.model((u, i))

                loss = criterion(output, target)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")
        
        self.rating_matrix = self.rating_matrix.to('cpu')
        self.model = self.model.to('cpu')

    def fit(self):
        """
        If does not use precomputed similarity matrix, calculate similarity matrix between items and store it in self.similarity_matrix. Then train the model
        """
        if self.similarity_matrix is None:
            self.calculate_similarity_matrix()
        self.train_model()

        
    def predict_rating(self, user_id, movie_id):
        """
        Predict rating of user_id for movie_id
        """
        user_index = self.user_to_index.get(user_id)
        movie_index = self.movie_to_index.get(movie_id)

        if user_index is None and movie_index is None:
            return torch.nanmean(self.rating_matrix)
        elif movie_index is None:
            return torch.nanmean(self.rating_matrix[user_index, :])
        elif user_index is None:
            return torch.nanmean(self.rating_matrix[:, movie_index])
        elif not torch.isnan(self.rating_matrix[user_index, movie_index]):
            return self.rating_matrix[user_index, movie_index]

        # rating_prediction = self.model((torch.tensor([user_index]), torch.tensor([movie_index]))).item()
        user_ratings = self.rating_matrix[user_index, :]
        movies_rated_by_user_index = torch.nonzero(~torch.isnan(user_ratings)).squeeze(dim=1)
        similarity_with_movie_index = self.similarity_matrix[movie_index]
        sorted_most_similar_with_movie_index = torch.argsort(similarity_with_movie_index, descending=True)
        qtu = torch.tensor([index for index in sorted_most_similar_with_movie_index if index in movies_rated_by_user_index])
        
        wjt = self.model.weight[qtu, movie_index]
        ruj = self.rating_matrix[user_index, qtu]
        
        rating_prediction = self.model.b_user[user_index] + self.model.b_item[movie_index] + torch.dot(wjt, ruj - self.model.b_user[user_index] - self.model.b_item[qtu]) / self.neighborhood_size
        rating_prediction = rating_prediction.item()
        
        return min(max(rating_prediction, 1), 5)