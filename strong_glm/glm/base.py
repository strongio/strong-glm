from typing import Type, Optional, Sequence, Callable, Union, Dict, Tuple, Iterator
from warnings import warn

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import FitFailedWarning

from skorch import NeuralNet
from skorch.callbacks import Callback, EarlyStopping
from skorch.dataset import CVSplit
from skorch.helper import SliceDict
from skorch.utils import to_numpy
from torch.distributions import Distribution, constraints, NegativeBinomial
from torch.optim import LBFGS

from strong_glm.hessian import hessian
from strong_glm.utils import to_tensor
from strong_glm.glm.utils import MultiOutputModule
from strong_glm.log_prob_criterion import NegLogProbLoss

import torch


class _NoBatchDataset:
    def __init__(self, *args, **kwargs):
        self.args = args
        if kwargs:
            warn(f"Unexpected kwargs {set(kwargs)}")

    def __len__(self):
        return 1


class _PassThruDataLoader:
    def __init__(self, dataset: _NoBatchDataset, batch_size: int = 1, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        if kwargs:
            warn(f"Unexpected kwargs {set(kwargs)}")

    def __iter__(self) -> Iterator[Tuple]:
        yield self.dataset.args


class Glm(NeuralNet):
    """Glm base class

    For fitting generalized linear models using a sklearn-style interface:

    ```
    glm = Glm(distribution=NegativeBinomial)
    ```
    """
    criterion_cls = NegLogProbLoss

    def __init__(self,
                 distribution: Type[Distribution],
                 lr: float = .25,
                 module: Optional[Type[torch.nn.Module]] = None,
                 optimizer: torch.optim.Optimizer = LBFGS,
                 early_stopping: tuple = (1e-5, 3),
                 distribution_param_names: Optional[Sequence[str]] = None,
                 max_epochs: int = 100,
                 batch_size: int = -1,
                 train_split: Optional[CVSplit] = None,
                 callbacks: Optional[Sequence[Callback]] = None,
                 criterion: Optional['Criterion'] = None,
                 iterator_train: 'DataLoader' = _PassThruDataLoader,
                 iterator_valid: 'DataLoader' = _PassThruDataLoader,
                 dataset: 'Dataset' = _NoBatchDataset,
                 **kwargs):
        """
        :param distribution: A torch.distributions.Distribution class.
        :param lr: Shortcut for optimizer__lr
        :param module: A torch.nn.Module. Instances will be created as sub-modules of MultiOutputModule.
        :param optimizer: The optimizer, default is LBFGS
        :param distribution_param_names: The names of the parameters of the distribution. Can usually be inferred.
        :param kwargs: Further keyword arguments that will be passed to skorch.NeuralNet
        """

        super().__init__(
            lr=lr,
            module=module,
            optimizer=optimizer,
            max_epochs=max_epochs,
            batch_size=batch_size,
            train_split=train_split,
            callbacks=callbacks,
            criterion=criterion,
            iterator_train=iterator_train,
            iterator_valid=iterator_valid,
            dataset=dataset,
            **kwargs
        )
        assert isinstance(distribution, type)
        self.distribution = distribution
        self.distribution_param_names = distribution_param_names

        self.module_input_feature_names_ = None
        self.laplace_params_ = None
        self.early_stopping = early_stopping

    def _get_params_for_optimizer(self, prefix, named_parameters):
        args, kwargs = super()._get_params_for_optimizer(prefix=prefix, named_parameters=named_parameters)
        # if issubclass(self.optimizer, LBFGS) or isinstance(self.optimizer, LBFGS):
        #     if 'line_search_fn' not in kwargs:
        #         kwargs['line_search_fn'] = 'strong_wolfe'
        return args, kwargs

    def initialize_optimizer(self, triggered_directly=True):
        super().initialize_optimizer()

    @property
    def distribution_param_names_(self):
        return list(self.module_.keys())

    @property
    def module_dtype_(self) -> torch.dtype:
        return next(self.module_.parameters()).dtype

    def infer(self, x: Union[torch.Tensor, SliceDict], **fit_params):
        x = to_tensor(x, device=self.device, dtype=self.module_dtype_)
        return super().infer(x=x, **fit_params)

    def predict(self, X: Union[torch.Tensor, SliceDict], type: str = 'mean', *args, **kwargs) -> np.ndarray:
        """
        Return an attribute of the distribution (by default the mean) as a numpy array.
        """
        X = to_tensor(X, device=self.device, dtype=self.module_dtype_)
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

    def predict_distribution(self, X: Union[torch.Tensor, SliceDict]) -> Distribution:
        params = self.infer(X)
        distribution_kwargs = dict(zip(self.distribution_param_names_, params))
        return self.distribution(**distribution_kwargs)

    def predict_proba(self, X: Union[torch.Tensor, SliceDict]):
        """
        Return the predicted probabilities, if applicable for this distribution (e.g. Binomial, Categorical).
        """
        return self.predict(X=X, type='probs')

    def predict_dataframe(self,
                          dataframe: 'DataFrame',
                          preprocessor: Union[ColumnTransformer, Sequence],
                          type: str,
                          x: torch.Tensor):
        """
        Experimental.
        """
        from pandas import DataFrame

        # check x, broadcast:
        x = to_tensor(x, device=self.device, dtype=self.module_dtype_)
        if len(x.shape) == 2:
            assert x.shape[1] == 1
        elif len(x.shape) == 1:
            x = x[:, None]
        else:
            raise RuntimeError("Expected `x` to be 1D")

        # generate dist-param predictions:
        assert len(dataframe.index) == len(set(dataframe.index))
        with torch.no_grad():
            X = to_tensor(preprocessor.transform(dataframe), device=self.device, dtype=self.module_dtype_)
            params = self.infer(X)
            distribution_kwargs = dict(zip(self.distribution_param_names_, params))
            dist = self.distribution(**distribution_kwargs)

            # plug x into method:
            pred_broadcasted = getattr(dist, type)(x)

        # flatten, joinable along original index:
        index_broadcasted = to_tensor(dataframe.index.values, device=self.device)[None, :].repeat(len(x), 1)
        x_broadcasted = x.repeat(1, len(dataframe.index))
        return DataFrame({
            'index': index_broadcasted.view(-1).numpy(),
            'x': x_broadcasted.view(-1).numpy(),
            type: pred_broadcasted.view(-1).numpy()
        })

    def get_loss(self,
                 y_pred: torch.Tensor,
                 y_true: torch.Tensor,
                 X: Optional[torch.Tensor] = None,
                 training: bool = False,
                 **kwargs):
        y_true = to_tensor(y_true, device=self.device, dtype=self.module_dtype_)
        neg_log_lik = self.criterion_(y_pred, y_true, **kwargs)
        penalty = self.criterion_.get_penalty(y_true=y_true, module=self.module_)
        return neg_log_lik + penalty

    @property
    def _default_callbacks(self):
        cbs = list(super()._default_callbacks)
        if self.early_stopping:
            threshold, patience = self.early_stopping
            cbs += [('early_stopping', EarlyStopping(monitor='train_loss', threshold=threshold, patience=patience))]
        return cbs

    def fit(self,
            X: torch.Tensor,
            y: torch.Tensor = None,
            input_feature_names: Optional[Sequence[str]] = None,
            **fit_params):
        """
        :param X: A model-matrix of predictors (should not include a constant term).
        :param y: A vector of outcomes.
        :param input_feature_names: The column-names of X.
        :param fit_params: Other kwargs passed to super
        :return: This instance
        """
        # infer number of input features if appropriate:
        if self.module_input_feature_names_ is None:
            self.module_input_feature_names_ = self._infer_input_feature_names(X, input_feature_names)

        ret_self = super().fit(X=X, y=y, **fit_params)
        any_nan_params = any(torch.isnan(param).any() for pname, param in self.module_.named_parameters())
        if any_nan_params:
            FitFailedWarning("Fitting resulted in `nan` parameters.")

        return ret_self

    def partial_fit(self, X, y=None, classes=None, input_feature_names: Optional[Sequence[str]] = None, **fit_params):
        # infer number of input features if appropriate:
        if self.module_input_feature_names_ is None:
            self.module_input_feature_names_ = self._infer_input_feature_names(X, input_feature_names)

        # convert y to the right dtype:
        y = to_tensor(y, device=self.device, dtype=self.module_dtype_)

        return super().partial_fit(X=X, y=y, classes=classes, **fit_params)

    def initialize_criterion(self):
        if self.criterion is None:
            self.criterion = self.criterion_cls
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
            elif self.distribution in _distribution_to_param_names:
                kwargs['names'] = _distribution_to_param_names[self.distribution]
            else:
                try:
                    kwargs['names'] = list(self.distribution.arg_constraints.keys())
                except AttributeError as e:
                    raise RuntimeError("Must supply distribution_param_names, unable to infer.") from e

        # if using default sub-module, can automatically set `in_features` keyword-arg:
        if MultiOutputModule.is_default_submodule(self.module):
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
            try:
                constraint = self.distribution.arg_constraints[param]
            except AttributeError:
                raise RuntimeError(f"`{self.distribution}` doesn't have an arg-constraints attribute; this is needed.")
            except KeyError:
                raise RuntimeError(
                    f"`{self.distribution.__name__}.arg_constraints` doesn't have param {param};"
                    f"\n{self.distribution.arg_constraints}"
                )
            ilink = _constraint_to_ilink.get(_constraint_hash(constraint), None)
            if ilink is None:
                # TODO: let the user override
                ilink = identity
                warn(
                    "distribution.arg_constraints['{}'] returned {}; unsure of proper inverse-link. Will use identity.".
                        format(param, constraint)
                )
            ilinks[param] = ilink
        return ilinks

    def estimate_laplace_params(self, X, y, **fit_params):
        means = torch.cat([param.data.view(-1) for param in self.module_.parameters()])

        # get loss, hessian:
        y_pred = self.infer(X, **fit_params)
        y_true = to_tensor(y, device=self.device, dtype=self.module_dtype_)
        loss = self.get_loss(y_pred, y_true, reduction='sum')
        hess = hessian(output=loss, inputs=list(self.module_.parameters()), allow_unused=True, progress=False)

        # create mvnorm for laplace approx:
        try:
            self.laplace_params_ = torch.distributions.MultivariateNormal(means, covariance_matrix=torch.inverse(hess))
            self.converged_ = True
        except RuntimeError as e:
            if 'lapack' in str(e) or 'cholesky' in str(e):
                warn("Model failed to converge; `laplace_params` cannot be estimated")
                fake_cov = (2 * means.abs().max() * torch.eye(len(hess))) ** 2
                self.laplace_params_ = torch.distributions.MultivariateNormal(means, covariance_matrix=fake_cov)
                self.converged_ = False
            else:
                raise e

    def summarize_laplace_params(self) -> 'DataFrame':
        if self.laplace_params_ is None:
            raise RuntimeError("Must run `estimate_laplace_params` first.")
        from pandas import DataFrame

        if not MultiOutputModule.is_default_submodule(self.module):
            raise NotImplementedError("Cannot run `summarize_laplace_params` if the default `module` was not used.")

        names = []
        for param_name, param_tens in self.module_.named_parameters():
            dist_param, w_or_b = param_name.split(".")
            if not param_tens.numel():
                continue
            if w_or_b == 'bias':
                names.append((dist_param, 'bias'))
            else:
                if isinstance(self.module_input_feature_names_, dict):
                    names.extend((dist_param, x) for x in self.module_input_feature_names_[dist_param])
                else:
                    names.extend((dist_param, x) for x in self.module_input_feature_names_)

        df = DataFrame({
            'estimate': self.laplace_params_.mean.numpy(),
            'std': self.laplace_params_.covariance_matrix.diagonal().sqrt().numpy()
        })
        df['dist_param'], df['feature'] = zip(*names)
        return df


def identity(x):
    return x


def _constraint_hash(constraint: constraints.Constraint) -> int:
    assert isinstance(constraint, constraints.Constraint)
    out = hash(type(constraint))
    out ^= hash(frozenset(constraint.__dict__.items()))
    return out


# useful if the distribution's init has multiple ways of specifying (e.g. both logits or probs)
_distribution_to_param_names = {
    NegativeBinomial: ['probs', 'total_count']
}

# torch.distributions has a whole 'transforms' module but I don't know if they provide a mapping
_constraint_to_ilink = {
    _constraint_hash(constraints.positive): torch.exp,
    _constraint_hash(constraints.greater_than(0)): torch.exp,
    _constraint_hash(constraints.unit_interval): torch.sigmoid,
    _constraint_hash(constraints.real): identity,
    # TODO: is there a way to make these work?
    _constraint_hash(constraints.greater_than_eq(0)): torch.exp,
    _constraint_hash(constraints.half_open_interval(0, 1)): torch.sigmoid,
    # TODO: constraints.interval
}
