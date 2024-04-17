import numpy as np
import torch
from recommenders.recommender_system import RecommenderSystem

class MemoryBasedRecommenderSystem(RecommenderSystem):
    def __init__(self, ratings_df, neighborhood_size, precompute_rating_matrix_path=None, precompute_similarity_matrix_path=None, similarity_measure='adjusted_cosine'):
        super().__init__(ratings_df, precompute_rating_matrix_path)
        self.neighborhood_size = neighborhood_size
        self.similarity_measure = similarity_measure

        # If precompute_similarity_matrix_path is provided, load the similarity matrix from the file
        if precompute_similarity_matrix_path is not None:
            self.similarity_matrix = np.load(precompute_similarity_matrix_path)
            self.similarity_matrix = torch.from_numpy(self.similarity_matrix).float() # Convert to torch tensor
        else:
            self.similarity_matrix = None