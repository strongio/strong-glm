from typing import Type, Optional, Sequence, Callable, Union, Dict
from warnings import warn

import torch
import numpy as np
from sklearn.compose import ColumnTransformer

from skorch import NeuralNet
from skorch.callbacks import Callback, EarlyStopping
from skorch.dataset import CVSplit
from skorch.helper import SliceDict
from skorch.utils import to_numpy
from torch.distributions import Distribution, constraints
from torch.optim.lbfgs import LBFGS

from strong_glm.glm.util import MultiOutputModule, SimpleLinear, to_tensor
from strong_glm.log_prob_criterion import NegLogProbLoss


class Glm(NeuralNet):
    criterion_cls = NegLogProbLoss

    def __init__(self,
                 distribution: Type[Distribution],
                 lr: float = .05,
                 module: Optional[Type[torch.nn.Module]] = None,
                 optimizer: torch.optim.Optimizer = LBFGS,
                 distribution_param_names: Optional[Sequence[str]] = None,
                 max_epochs: int = 100,
                 batch_size: int = -1,
                 train_split: Optional[CVSplit] = None,
                 callbacks: Optional[Sequence[Callback]] = None,
                 **kwargs):
        """
        :param distribution: A torch.distributions.Distribution, generally one with positive support.
        :param lr: Shortcut for optimizer__lr
        :param module: A torch.nn.Module. Instances will be created as sub-modules of MultiOutputModule.
        :param optimizer: The optimizer, default is LBFGS
        :param distribution_param_names: The names of the parameters of the distribution. Can usually be inferred.
        :param kwargs: Further keyword arguments that will be passed to skorch.NeuralNet
        """

        super().__init__(
            lr=lr,
            module=module,
            criterion=self.criterion_cls,
            optimizer=optimizer,
            max_epochs=max_epochs,
            batch_size=batch_size,
            train_split=train_split,
            callbacks=callbacks,
            **kwargs
        )

        self.distribution = distribution
        self.distribution_param_names = distribution_param_names

        self.module_input_feature_names_ = None

    @property
    def distribution_param_names_(self):
        return list(self.module_.keys())

    @property
    def module_dtype(self) -> torch.dtype:
        return next(self.module_.parameters()).dtype

    def infer(self, x: Union[torch.Tensor, SliceDict], **fit_params):
        x = to_tensor(x, device=self.device, dtype=self.module_dtype)
        return super().infer(x=x, **fit_params)

    def predict(self, X: Union[torch.Tensor, SliceDict], type: str = 'mean', *args, **kwargs) -> np.ndarray:
        """
        Return an attribute of the distribution (by default the mean) as a numpy array.
        """
        y_out = []
        for params in self.forward_iter(X, training=False):
            batch_size = len(params[0])
            distribution_kwargs = dict(zip(self.distribution_param_names_, params))
            dist = self.distribution(**distribution_kwargs)
            yp = getattr(dist, type)
            if callable(yp):
                yp = yp(*args, **kwargs)
            yp = to_numpy(yp)
            if yp.shape[0] != batch_size:
                raise RuntimeError(
                    f"`{self.distribution.__name__}.{type}` produced a tensor whose leading dim is {yp.shape[0]}, "
                    f"expected {batch_size}."
                )
            y_out.append(yp)
        y_out = np.concatenate(y_out, 0)
        return y_out

    def predict_proba(self, X: Union[torch.Tensor, SliceDict]):
        """
        Return the predicted probabilities, if applicable for this distribution (e.g. Binomial, Categorical).
        """
        return self.predict(X=X, type='probs')

    def get_loss(self,
                 y_pred: torch.Tensor,
                 y_true: torch.Tensor,
                 X: Optional[torch.Tensor] = None,
                 training: bool = False):
        neg_log_lik = super().get_loss(y_pred, y_true)
        penalty = self.criterion_.get_penalty(y_true=y_true, **dict(self.module_.named_parameters()))
        return neg_log_lik + penalty

    @property
    def _default_callbacks(self):
        sup = list(super()._default_callbacks)
        return sup + [EarlyStopping(monitor='train_loss', threshold=1e-6)]

    def fit(self, X, y=None, input_feature_names: Optional[Sequence[str]] = None, **fit_params):
        # infer number of input features if appropriate:
        if self.module_input_feature_names_ is None:
            self.module_input_feature_names_ = self._infer_input_feature_names(X, input_feature_names)

        # TODO: if dataframe, X.values

        return super().fit(X=X, y=y, **fit_params)

    def partial_fit(self, X, y=None, classes=None, input_feature_names: Optional[Sequence[str]] = None, **fit_params):
        # infer number of input features if appropriate:
        if self.module_input_feature_names_ is None:
            self.module_input_feature_names_ = self._infer_input_feature_names(X, input_feature_names)

        # convert y to the right dtype:
        y = to_tensor(y, device=self.device, dtype=self.module_dtype)

        return super().partial_fit(X=X, y=y, classes=classes, **fit_params)

    def initialize_criterion(self):
        if getattr(self, 'module_', None) is None:
            # defer because we use the initialized module to get `self.distribution_param_names_`. Instead,
            # initialize_criterion will be called at the end of `initialize_module`.
            return self
        criterion_params = self._get_params_for('criterion')

        # initialize:
        self.criterion_ = self.criterion(
            param_names=self.distribution_param_names_,
            distribution=self.distribution,
            **criterion_params
        )
        self.criterion_ = self.criterion_.to(self.device)
        return self

    def initialize_module(self):
        kwargs = self._get_params_for('module')
        kwargs = self._infer_module_kwargs(kwargs)

        # re-initializing?
        if self.initialized_ and self.verbose:
            print(self._format_reinit_msg("module", kwargs))

        # initialize:
        module = MultiOutputModule(**kwargs)
        self.module_ = module.to(self.device)

        # we deferred this previously b/c it required the module
        self.initialize_criterion()

        return self

    def _infer_input_feature_names(self,
                                   X: Union[torch.Tensor, Dict[str, torch.Tensor]],
                                   input_feature_names: Optional[Sequence[str]]) -> Sequence[str]:
        if input_feature_names is not None:
            if isinstance(X, dict):
                if isinstance(input_feature_names, dict):
                    return input_feature_names
                else:
                    return {k: input_feature_names for k in X.keys()}
            else:
                if isinstance(input_feature_names, dict):
                    raise ValueError("Got {} for `input_feature_names` but `{}` for X.".format(input_feature_names, X))
                else:
                    return input_feature_names

        if isinstance(X, dict):
            return {k: self._infer_input_feature_names(v, None) for k, v in X.items()}
        elif hasattr(X, 'columns'):
            return list(X.columns)
        else:
            if self.verbose:
                print("Consider passing `input_feature_names` for named features.")
            return ["x{}".format(i) for i in range(X.shape[1])]

    def _infer_module_kwargs(self, kwargs: Dict) -> Dict:
        kwargs = kwargs.copy()

        # if output names not supplied, infer:
        if not kwargs.get('names'):
            if self.distribution_param_names:
                kwargs['names'] = self.distribution_param_names
            else:
                try:
                    kwargs['names'] = list(self.distribution.arg_constraints.keys())
                except AttributeError as e:
                    raise RuntimeError("Must supply distribution_param_names, unable to infer.") from e

        # if using default sub-module, can automatically set `in_features` keyword-arg:
        if issubclass(self.module or SimpleLinear, SimpleLinear):
            if isinstance(self.module_input_feature_names_, dict):
                kwargs.update(
                    {f'{k}__in_features': len(self.module_input_feature_names_.get(k, [])) for k in kwargs['names']}
                )
            else:
                kwargs["in_features"] = len(self.module_input_feature_names_)

        # if inv-links not supplied, infer:
        if not kwargs.get('inv_links'):
            kwargs['inv_links'] = self._infer_ilinks(kwargs['names'])

        # `module` passed at init must get wrapped in MultiOutputModule
        assert 'sub_module_cls' not in kwargs
        kwargs['sub_module_cls'] = self.module

        return kwargs

    def _infer_ilinks(self, param_names: Sequence[str]) -> Dict[str, Callable]:
        ilinks = {}
        for param in param_names:
            constraint = getattr(self.distribution, 'arg_constraints', {}).get(param, None)
            ilink = _constraint_to_ilink.get(constraint, None)
            if ilink is None:
                ilink = identity
                warn(
                    "distribution.arg_constraints[{}] returned {}; unsure of proper inverse-link. Will use identity.".
                        format(param, constraint)
                )
            ilinks[param] = ilink
        return ilinks


def identity(x):
    return x


_constraint_to_ilink = {
    constraints.positive: torch.exp,
    constraints.unit_interval: torch.sigmoid,
    constraints.real: identity
}

# def laplace_params(self, y_pred, y_true, input_features=None):
#     input_features = input_features or self.input_features_
#     means = None  # TODO
#     if len(means) != len(input_features):
#         warn("TODO: coherent warning")
#         feature_names = ["x{}".format(i) for i in range(len(means))]
#
#     # get hessian:
#     loss = self.get_loss(y_pred, y_true)
#     hess = hessian(output=loss, inputs=list(self.module_.parameters()), allow_unused=True, progress=False)
#
#     # create mvnorm for laplace approx:
#     try:
#         self._param_estimates = torch.distributions.MultivariateNormal(means, torch.inverse(hess))
#     except RuntimeError as e:
#         if 'Lapack' in e.args[0]:
#             warn(f"Hessian was not invertible, model may not have converged. Returning ~0 cov. Hessian:\n{hess}")
#             cov = torch.eye(hess.shape[0]) * 10e-8
#             self._param_estimates = torch.distributions.MultivariateNormal(means, cov)
#         else:
#             raise e
