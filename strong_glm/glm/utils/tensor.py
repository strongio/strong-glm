from typing import Union, Optional

import torch
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
