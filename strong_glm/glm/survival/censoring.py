from typing import Tuple, Optional

import numpy as np
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES


def cens_y_to_indicator(time: np.ndarray, is_upper_cens: np.ndarray, window: float) -> np.ndarray:
    time = np.asanyarray(time)
    is_upper_cens = np.asanyarray(is_upper_cens)

    out = np.zeros_like(time)

    # possibilities:
    # - happened, before window
    out[(is_upper_cens == 0) & (time <= window)] = 1.0
    # - happened, after window
    out[(is_upper_cens == 0) & (time > window)] = 0.0
    # - not happened in original dataset, but that's smaller than `window`
    out[(is_upper_cens == 1) & (time < window)] = float('nan')
    # - not happened in original dataset, even though we 'waited' >= `window`
    out[(is_upper_cens == 1) & (time > window)] = 0.0

    return out


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


def km_summary(time: np.ndarray,
               is_upper_cens: np.ndarray,
               lower_trunc: Optional[np.ndarray] = None) -> 'DataFrame':
    from pandas import DataFrame

    time = np.asanyarray(time)
    is_upper_cens = np.asanyarray(is_upper_cens)
    if lower_trunc is not None:
        lower_trunc = np.asanyarray(lower_trunc)

    if lower_trunc is None:
        lower_trunc = np.zeros_like(time)
    assert len(lower_trunc) == len(time)
    if not (lower_trunc <= time).all():
        raise ValueError("All of `lower_trunc` should be <= `time`.")
    sorted_times = np.unique(np.concatenate([time, lower_trunc]))

    df = {'time': [], 'num_at_risk': [], 'num_events': [], 'km_estimate': []}

    survival = 1.0
    num_at_risk = 0
    for t in sorted_times:
        num_at_risk += (t == lower_trunc).sum()
        is_time_bool = (t == time)
        num_censored = np.sum(is_upper_cens[is_time_bool]).item()
        num_events = np.sum(is_time_bool).item() - num_censored
        survival *= (1. - num_events / num_at_risk)

        df['time'].append(t.item())
        df['num_at_risk'].append(num_at_risk)
        df['num_events'].append(num_events)
        df['km_estimate'].append(survival)

        # for next iter:
        num_at_risk -= (num_events + num_censored)

    return DataFrame(df)
