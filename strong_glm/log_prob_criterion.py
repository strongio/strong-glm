from typing import Sequence, Type, Optional

import torch
from skorch.helper import SliceDict
from torch.distributions import Distribution

from strong_glm.penalty import L2, SmoothL1

_reductions = {
    'mean': torch.mean,
    'sum': torch.sum
}


class NegLogProbLoss(torch.nn.modules.loss._Loss):
    """
    Given a torch.distribution type, give the negative log-loss from a network that outputs predicted parameters.
    """
    _penalty_aliases = {
        'l2': L2,
        'smooth_l1': SmoothL1,
        'huber': SmoothL1
    }

    def __init__(self,
                 param_names: Sequence[str],
                 distribution: Type[Distribution],
                 reduction: str = 'mean',
                 penalty: float = 0.0,
                 penalty_type: str = 'l2'):
        super().__init__(reduction=reduction)
        self.param_names = param_names
        self.distribution = distribution

        if isinstance(penalty_type, str):
            penalty_type = self._penalty_aliases[penalty_type]
        if isinstance(penalty, int):
            penalty = float(penalty)
        self.penalty = penalty_type(penalty)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, reduction: Optional[str] = None) -> torch.Tensor:
        """
        :param y_pred: Tuple of predicted parameters.
        :param y_true: Tensor of the observed target.
        :param reduction: For overriding self.reduction.
        :return: The penalty
        """

        distribution_kwargs = SliceDict()
        for p_name, p_pred in zip(self.param_names, y_pred):
            if len(p_pred.shape) == 1 and len(y_true.shape) == 2:
                assert y_true.shape[-1] == 1
                p_pred = p_pred.unsqueeze(-1)
            if len(p_pred.shape) == 2 and len(y_true.shape) == 1:
                assert p_pred.shape[-1] == 1
                y_true = y_true.unsqueeze(-1)

            assert y_true.shape == p_pred.shape
            distribution_kwargs[p_name] = p_pred
        valid = ~torch.isnan(y_true)
        neg_log_probs = -self.distribution(**distribution_kwargs[valid]).log_prob(y_true[valid])
        if not valid.all():
            pass  # TODO: warn?

        reduction = reduction or self.reduction
        return _reductions[reduction](neg_log_probs)

    def get_penalty(self, y_true: torch.Tensor, module, reduction: Optional[str] = None):
        """
        :param y_true: Tensor of the observed target.
        :param reduction: For overriding self.reduction.
        :param kwargs: The parameters, retrieved via `dict(module.named_parameters())`
        :return: The penalty
        """
        assert isinstance(y_true, (torch.Tensor, SliceDict))
        penalty = self.penalty(module)
        reduction = reduction or self.reduction
        if reduction == 'mean':
            penalty = penalty / len(y_true)
        return penalty
