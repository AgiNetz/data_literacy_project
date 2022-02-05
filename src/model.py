from sklearn.model_selection import train_test_split


def get_splits(data, test_size, random_seed=None):
    data = (data,) if not isinstance(data, tuple) else data
    return train_test_split(*data, test_size=test_size, random_state=random_seed)
