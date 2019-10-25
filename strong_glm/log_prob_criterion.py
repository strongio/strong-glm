from typing import Sequence, Type, Optional, Callable

import torch
from skorch.helper import SliceDict
from torch.distributions import Distribution

from strong_glm.penalty.base import Penalty
from strong_glm.penalty.penalties import NoPenalty, L2

_reductions = {
    'mean': torch.mean,
    'sum': torch.sum
}


class NegLogProbLoss(torch.nn.modules.loss._Loss):
    """
    Given a torch.distribution type, give the negative log-loss from a network that outputs predicted parameters.
    """

    def __init__(self,
                 param_names: Sequence[str],
                 distribution: Type[Distribution],
                 reduction: str = 'mean',
                 penalty: Optional[float] = None):
        super().__init__(reduction=reduction)
        self.param_names = param_names
        self.distribution = distribution

        if penalty is None:
            self.penalty = NoPenalty()
        elif isinstance(penalty, float):
            self.penalty = L2(multi=penalty)
        elif isinstance(penalty, (Penalty, Callable)):
            self.penalty = penalty
        elif isinstance(penalty, dict):
            raise NotImplementedError("TODO: different penalty for each `param_names`")
        else:
            raise ValueError(f"Expected `penalty` to be a float for L2-penalty or a `Penalty` instance. Got: {penalty}")

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        neg_log_probs = -self.distribution(*y_pred).log_prob(y_true)
        return _reductions[self.reduction](neg_log_probs)

    def get_penalty(self, y_true: torch.Tensor, **kwargs):
        assert isinstance(y_true, (torch.Tensor, SliceDict))
        penalty = self.penalty(**kwargs)
        if self.reduction == 'mean':
            penalty = penalty / len(y_true)
        return penalty
