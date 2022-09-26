"""
To add a model, create a new directory in bayesian_benchmarks.models (here) and make file models.py containing at
least one of the classes below.

Model usage is similar to sklearn. For regression:

model = RegressionModel(is_test=False)
model.fit(X, Y)
mean, var = model.predict(X_test)  # first two moments of the posterior
samples = model.sample(X_test, S)  # samples from the posterior

For classification:
model = ClassificationModel(K, is_test=False)  # K is the number of classes
model.fit(X, Y)
p = model.predict(X_test)          # predictive probabilities for each class (i.e. onehot)

It should be feasible to call fit and predict many times (e.g. avoid rebuilding a tensorflow graph on each call).

"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


class RegressionModel(ABC):
    @abstractmethod
    def __init__(self, is_test: bool = False, seed: int = 0) -> None:
        """
        :param is_test: whether to run quickly for testing purposes
        """
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Train the model (and probably create the model, too, since there is no
        shape information on the __init__)

        :param X: numpy array, of shape [N, Dx]
        :param Y: numpy array, of shape [N, Dy]
        """
        pass

    @abstractmethod
    def predict(self, Xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        The predictive mean and variance

        :param Xs: numpy array, of shape [N, Dx]
        :return: mean, var, both of shape [N, Dy]
        """
        pass

    @abstractmethod
    def sample(self, Xs: np.ndarray, S: int) -> np.ndarray:
        """
        Samples from the posterior
        :param Xs: numpy array, of shape [N, Dx]
        :param S: number of samples
        :return: numpy array, of shape [S, N, Dy]
        """
        pass


class ClassificationModel(ABC):
    @abstractmethod
    def __init__(self, K: int, is_test: bool = False, seed: int = 0) -> None:
        """
        :param K: number of classes
        :param is_test: whether to run quickly for testing purposes
        """

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Train the model (and probably create the model, too, since there is no
        shape information on the __init__)

        Note Y is not onehot, but is an int array of labels in {0, 1, ..., K-1}

        :param X: numpy array, of shape [N, Dx]
        :param Y: numpy array, of shape [N, 1]
        """
        pass

    @abstractmethod
    def predict(self, Xs: np.ndarray) -> np.ndarray:
        """
        The predictive probabilities

        :param Xs: numpy array, of shape [N, Dx]
        :return: p, of shape [N, K]
        """
        pass
