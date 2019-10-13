from typing import Tuple

import torch

from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES


def unpack_cens_y(y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert len(y.shape) == 2
    if y.shape[1] <= 1:
        raise ValueError(
            "Expected `y` to have at least two columns: the first with values the second with censoring indicators."
        )
    else:
        y_cens = y[:, 1]
        if not ((y_cens == 0.) | (y_cens == 1.)).all():
            raise ValueError("The second column of `y` should be a censoring indicator, with only 0s and 1s")
        y_vals = y[:, 0]
        if y.shape[1] == 3:
            y_ltrunc = y[:, 2]
        else:
            y_ltrunc = torch.zeros_like(y_vals)
    return y_vals, y_cens, y_ltrunc


class CensScaler(StandardScaler):
    """
    Scaler for arrays representing right-censored time-to-event data.
    """

    def __init__(self):
        super().__init__(with_mean=True, with_std=False, copy=True)

    def partial_fit(self, X, y=None):
        return super().partial_fit(X=X[:, [0]])

    def transform(self, X, copy=None):
        # copy:
        X = check_array(X,
                        accept_sparse=False,
                        copy=self.copy,
                        estimator=self,
                        dtype=FLOAT_DTYPES,
                        force_all_finite='allow-nan')

        num_cols = X.shape[1]
        assert 1 <= num_cols <= 3

        X[:, [0]] = X[:, [0]] / self.mean_
        if num_cols == 3:
            X[:, [2]] = X[:, [2]] / self.mean_
        return X

    def inverse_transform(self, X, copy=None):
        # copy:
        X = check_array(X,
                        accept_sparse=False,
                        copy=self.copy,
                        estimator=self,
                        dtype=FLOAT_DTYPES,
                        force_all_finite='allow-nan')

        num_cols = X.shape[1]
        assert 1 <= num_cols <= 3

        X[:, [0]] = X[:, [0]] * self.mean_
        if num_cols == 3:
            X[:, [2]] = X[:, [2]] * self.mean_
        return X
