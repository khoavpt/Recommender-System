import pandas as pd
from sklearn.model_selection import train_test_split

def get_data(data_path, train_size, random_state=42):
    """
    Args:
        data_path: path to ratings.csv file
        train_size: size of training set over full dataset
        random_state: random state for train_test_split
    Returns:
        train_set: training set
        test_set: test set
    """
    ratings_df = pd.read_csv(data_path, usecols=['userId', 'movieId', 'rating'])
    train_set, test_set = train_test_split(ratings_df, test_size=1-train_size, random_state=random_state)
    return train_set, test_set