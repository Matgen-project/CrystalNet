from typing import Any, List

import numpy as np


class StandardScaler:
    """A StandardScaler normalizes a dataset.

    When fit on a dataset, the StandardScaler learns the mean and standard deviation across the 0th axis.
    When transforming a dataset, the StandardScaler subtracts the means and divides by the standard deviations.
    """

    def __init__(self, means: np.ndarray = None, stds: np.ndarray = None, replace_nan_token: Any = None):
        """
        Initialize StandardScaler, optionally with means and standard deviations precomputed.

        :param means: An optional 1D numpy array of precomputed means.
        :param stds: An optional 1D numpy array of precomputed standard deviations.
        :param replace_nan_token: The token to use in place of nans.
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, x: List[List[float]]) -> 'StandardScaler':
        """
        Learns means and standard deviations across the 0th axis.

        :param x: A list of lists of floats.
        :return: The fitted StandardScaler.
        """
        x = np.array(x).astype(float)
        self.means = np.nanmean(x, axis=0)
        self.stds = np.nanstd(x, axis=0)
        self.means = np.where(np.isnan(self.means), np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds), np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(self.stds.shape), self.stds)

        return self

    def transform(self, x: List[List[float]]):
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.

        :param x: A list of lists of floats.
        :return: The transformed data.
        """
        x = np.array(x).astype(float)
        transformed_with_nan = (x - self.means) / self.stds
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

    def inverse_transform(self, x: List[List[float]]):
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.

        :param x: A list of lists of floats.
        :return: The inverse transformed data.
        """
        x = np.array(x).astype(float)
        transformed_with_nan = x * self.stds + self.means
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none
