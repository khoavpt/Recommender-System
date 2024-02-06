from recommenders.recommender_system import RecommenderSystem

class MemoryBasedRecommenderSystem(RecommenderSystem):
    def __init__(self, ratings_df, model_size, similarity_measure='adjusted_cosine'):
        super().__init__(ratings_df)
        self.similarity_matrix = None
        self.model_size = model_size
        self.similarity_measure = similarity_measure