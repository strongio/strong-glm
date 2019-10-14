from typing import Sequence, Dict, Callable, Optional, Type, Tuple, Union

import torch
from torch.distributions.utils import broadcast_all

from skorch.helper import SliceDict
from skorch.utils import to_tensor as _to_tensor_base, is_pandas_ndframe


def to_tensor(X: Union[torch.Tensor, SliceDict],
              device: Union[torch.device, str],
              dtype: Optional[torch.dtype] = None) -> Union[torch.Tensor, SliceDict]:
    """
    Behaves slightly differently than skorch's to_tensor: (a) DataFrames simply get their `values` extracted, (b)
    supports dtype conversion.
    """
    if isinstance(X, dict):
        return SliceDict(**{k: to_tensor(v, device=device, dtype=dtype) for k, v in X.items()})
    elif isinstance(X, (list, tuple)):
        return type(X)(to_tensor(v, device=device, dtype=dtype) for v in X)
    else:
        if is_pandas_ndframe(X):
            X = X.values
        tensor = _to_tensor_base(X=X, device=device)

    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor


class SimpleLinear(torch.nn.Linear):
    """
    Similar to torch.nn.Linear but (a) generalizes to intercept-only (bias-only) model, (b) uses only lightly jittered
    inits.
    """
    init_std_dev = .01

    def reset_parameters(self):
        if self.bias is not None:
            self.bias.data = self.init_std_dev * torch.randn_like(self.bias.data)
        self.weight.data = self.init_std_dev * torch.randn_like(self.weight.data)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.in_features:
            return self.bias.clone()
        return super().forward(input)


class MultiOutputModule(torch.nn.ModuleDict):
    """
    Wrapper around torch.nn.Modules to help generated named predictions. This generally used by `SurvivalModel` for
    predicting the params of a probability distribution.

    Example:

    ```
    my_module = MultiOutputModule(
        names=['scale','shape'], inv_links=[torch.exp, torch.exp], scale__in_features=2, shape__in_features=1
    )
    ```
    """

    def __init__(self,
                 names: Sequence[str],
                 inv_links: Dict[str, Callable],
                 sub_module_cls: Optional[Type[torch.nn.Module]] = None,
                 **kwargs):
        """
        :param names: The names of what is being predicted. The output of forward will be a tuple with predictions for
        each in this ordering.
        :param inv_links: The inverse-links to apply to the outputs of the sub-modules.
        :param sub_module_cls: A nn.Module type. For each `name` in `names`, one will be instantiated. If unspecified,
        will use `Regression` (similar to torch.nn.Linear).
        :param kwargs: Further arguments that will be passed each time `sub_module_cls.__init__` is called. If different
         kwargs are needed for each, can specify using the `name` from `names` as in: `{name}__{kwarg}`.
        """
        self.inv_links = inv_links
        kwargs, per_output_kwargs = self._organize_kwargs(kwargs, names)
        if sub_module_cls is None:
            sub_module_cls = SimpleLinear
            kwargs['out_features'] = 1
        super().__init__([(nm, sub_module_cls(**kwargs, **per_output_kwargs[nm])) for nm in names])

    @staticmethod
    def _organize_kwargs(kwargs: Dict, names: Sequence[str]) -> Tuple[Dict, Dict]:
        per_output_kwargs = {nm: {} for nm in names}
        for key in list(kwargs.keys()):
            parent, _, child = key.partition("__")
            if child:
                if parent in per_output_kwargs:
                    per_output_kwargs[parent][child] = kwargs.pop(key)
                else:
                    raise ValueError(f"Key `{key}` uses __ separator, but `{parent}` not in `{names}`")
        return kwargs, per_output_kwargs

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        input = self._validate_input(*args, **kwargs)

        y_pred = []
        for nm, sub_module in self.items():
            sub_input = input.get(nm, None)
            sub_y_pred = self.inv_links[nm](sub_module(sub_input))
            y_pred.append(self._validate_pred_shape(sub_y_pred, nm))

        return tuple(broadcast_all(*y_pred))

    def _validate_input(self, *args, **kwargs):
        if args:
            if kwargs or len(args) > 1:
                raise RuntimeError(
                    "{} expects either a single unnamed argument or keyword-arguments, got: {}.".
                        format(self.__class__.__name__, (args, kwargs))
                )
            input = args[0]
        elif kwargs:
            input = kwargs
        else:
            raise TypeError("`{}` got no input arguments.".format(self.__class__.__name__))
        if isinstance(input, torch.Tensor):
            input = SliceDict(**{k: input for k in self.keys()})
        return input

    @staticmethod
    def _validate_pred_shape(x: torch.Tensor, nm: str):
        if len(x.shape) == 2:
            x = x.squeeze(-1)
        if len(x.shape) > 1:
            raise RuntimeError(f"The output for `{nm}` has invalid shape {x.shape}, expected 1D.")
        return x
