import numpy as np

def cosine_similarity(vector1, vector2):
    """Cosine similarity between two vectors."""
    denominator = (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    return np.dot(vector1, vector2) / denominator if denominator != 0 else 0

def adjusted_cosine_similarity(vector1, vector2):
    """Adjusted cosine similarity between two vectors."""
    if vector1.size == 0:
        return 0
    normalized_vector1 = vector1 - np.mean(vector1)
    normalized_vector2 = vector2 - np.mean(vector2)

    denominator = np.linalg.norm(normalized_vector1) * np.linalg.norm(normalized_vector2)

    return np.dot(normalized_vector1, normalized_vector2) / denominator if denominator != 0 else 0

def pearson_correlation(vector1, vector2):
    """Pearson correlation between two vectors."""
    raise NotImplementedError
