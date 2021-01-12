from typing import Sequence, Union, Optional, Type

import torch
from sklearn.compose import ColumnTransformer
from torch.distributions import Distribution

from strong_glm.glm.base import Glm, LBFGS
from strong_glm.glm.survival.censoring import CensScaler, km_summary
from strong_glm.glm.survival.loss import CensNegLogProbLoss
from strong_glm.utils import to_tensor

import numpy as np


class SurvivalGlm(Glm):
    criterion_cls = CensNegLogProbLoss

    def __init__(self,
                 distribution: Type[Distribution],
                 scale_y: bool,
                 lr: float = .05,
                 module: Optional[Type[torch.nn.Module]] = None,
                 optimizer: torch.optim.Optimizer = LBFGS,
                 distribution_param_names: Optional[Sequence[str]] = None,
                 **kwargs):

        self.scale_y = scale_y
        self.y_scaler_ = None

        super().__init__(
            distribution=distribution,
            lr=lr,
            module=module,
            optimizer=optimizer,
            distribution_param_names=distribution_param_names,
            **kwargs
        )

    def fit(self, X, y=None, **fit_params):
        # initialize/reset scaler:
        if self.scale_y:
            if not self.y_scaler_:
                self.y_scaler_ = CensScaler()
            self.y_scaler_._reset()

        return super().fit(X=X, y=y, **fit_params)

    def partial_fit(self, X, y=None, classes=None, **fit_params):
        # (partial_)fit scaler:
        if self.scale_y:
            if not self.y_scaler_:
                self.y_scaler_ = CensScaler()
            self.y_scaler_.partial_fit(y)
            y = self.y_scaler_.transform(y)

        return super().partial_fit(X=X, y=y, **fit_params)

    def predict(self, X: Union[torch.Tensor, 'SliceDict'], type: str = 'mean', *args, **kwargs) -> 'ndarray':
        if self.verbose and self.scale_y:
            print("Reminder: Model was fit w/scale_y=True, so predictions are after scaling. See `model.y_scaler_`")
        return super().predict(X=X, type=type, *args, **kwargs)

    def km_summary(self,
                   dataframe: 'DataFrame',
                   preprocessor: Union[ColumnTransformer, Sequence],
                   time_colname: str,
                   censor_colname: str,
                   start_time_colname: Optional[str] = None) -> 'DataFrame':
        """
        :param dataframe: A pandas DataFrame, or a DataFrameGroupBy (i.e., the result of calling `df.groupby([...])`).
        :param preprocessor: Either a sklearn ColumnTransformer that takes the dataframe and returns X, or a list of
        column-names (such that `X = dataframe.loc[:,preprocessor].values`)
        :param time_colname: The column-name in the dataframe for time-to-event.
        :param censor_colname: The column-name in the dataframe for the censoring indicator.
        :param start_time_colname: Optional, the column-name in the dataframe for start-times (for left-truncation).
        :return: A DataFrame with kaplan-meier estimates.
        """
        try:
            from pandas.core.groupby.generic import DataFrameGroupBy
        except ImportError as e:
            raise ImportError("Must install pandas for `km_summary`") from e

        if isinstance(dataframe, DataFrameGroupBy):
            df_applied = dataframe.apply(
                self.km_summary,
                preprocessor=preprocessor,
                time_colname=time_colname,
                censor_colname=censor_colname,
                start_time_colname=start_time_colname
            )
            index_idx = [i for i, _ in enumerate(df_applied.index.names)]
            return df_applied.reset_index(level=index_idx[:-1], drop=False)
        else:

            # preprocess X:
            if hasattr(preprocessor, 'transform'):
                X = preprocessor.transform(dataframe)
            else:
                X = dataframe.loc[:, preprocessor].values
            X = to_tensor(X, device=self.device, dtype=self.module_dtype_)

            # km estimate:
            df_km = km_summary(
                time=dataframe[time_colname].values,
                is_upper_cens=dataframe[censor_colname],
                lower_trunc=dataframe[start_time_colname] if start_time_colname else None
            )

            # generate predicted params, transpose as inputs to distribution:
            with torch.no_grad():
                y_preds = self.infer(X)
                kwargs = {k: y_true[None, :] for k, y_true in zip(self.distribution_param_names_, y_preds)}
                distribution = self.distribution(**kwargs)

            # get unique times in distribution-friendly format:
            y = df_km.loc[:, ['time']].values
            if self.scale_y:
                y = self.y_scaler_.transform(y)
            y = to_tensor(y, device=self.device, dtype=self.module_dtype_)

            # b/c dist-kwargs transposed, broadcasting logic means we get array with dims: (times, dataframe_rows)
            observed = y[:, [0]]
            surv = 1. - distribution.cdf(observed)
            if start_time_colname:
                # TODO: either figure out if taking the average is valid, or emit a warning
                min_ltrunc = np.full_like(observed, fill_value=dataframe[start_time_colname].min())
                if self.scale_y:
                    min_ltrunc = self.y_scaler_.transform(min_ltrunc)
                min_ltrunc = to_tensor(min_ltrunc, device=self.device, dtype=self.module_dtype_)
                surv /= (1. - distribution.cdf(min_ltrunc))
            # this is then reduced, collapsing across dataframe rows, so that we get a mean estimate for this dataset
            df_km['model_estimate'] = torch.mean(surv, dim=1)
            return df_km

    def estimate_laplace_params(self, X, y, **fit_params):
        y = self.y_scaler_.transform(y)
        return super().estimate_laplace_params(X=X, y=y, **fit_params)
