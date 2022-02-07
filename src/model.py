from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
import numpy as np

import logging


def get_splits(data, test_size, random_seed=None):
    data = (data,) if not isinstance(data, tuple) else data
    return train_test_split(*data, test_size=test_size, random_state=random_seed)


def find_best_linear_model(train_data, labels):
    polynomial_range = np.linspace(1, 6, num=6)
    best = (np.NINF, {})
    for poly in polynomial_range:
        polynomial_fts = np.hstack([train_data**(i+1) for i in range(int(poly))])
        score = np.mean(cross_val_score(LinearRegression(fit_intercept=False), polynomial_fts, labels, cv=5))
        logging.info("Linear model search: Score: {s}, Polynomial: {p}".format(s=score, p=poly))
        if score > best[0]:
            best = (score, {"polynomial": poly})
    return best


def find_best_ridge_model(train_data, labels):
    polynomial_range = np.linspace(1, 6, num=6)
    alpha_range = np.linspace(0.1, 10.0, num=100)
    best = (np.NINF, {})
    for poly in polynomial_range:
        for a in alpha_range:
            polynomial_fts = np.hstack([train_data**(i+1) for i in range(int(poly))])
            score = np.mean(cross_val_score(Ridge(alpha=a), polynomial_fts, labels, cv=5))
            logging.info("Ridge model search: Score: {s}, Polynomial: {p}, alpha: {a}".format(s=score, p=poly, a=a))
            if score > best[0]:
                best = (score, {"polynomial": poly, "alpha": a})
    return best


def find_best_neural_model(train_data, labels, random_state):
    alpha_range = np.linspace(0.00001, 0.0005, num=10)
    sizes = [(5,), (5, 5), (10,), (10, 10), (15,), (15, 15), (20,), (20,20), (25,), (25, 25)]
    best = (np.NINF, {})
    for a in alpha_range:
        for size in sizes:
            score = np.mean(cross_val_score(MLPRegressor(solver='lbfgs', hidden_layer_sizes=size,
                                                         max_iter=2000, random_state=random_state),
                                            train_data, labels, cv=5))
            logging.info("Neural network search: Score: {s}, alpha: {a} size: {sz}".format(s=score, a=a, sz=size))
            if score > best[0]:
                best = (score, {"alpha": a, "size": size})
    return best
