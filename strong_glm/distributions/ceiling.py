import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all


class CeilingMixin:
    """
    Mixin for torch.distribution.Distributions where instead of assuming that events always happen eventually, the
    event-probability asymptotes to some probability less than 1.0.
    """

    def __init__(self, ceiling: torch.Tensor, *args, **kwargs):
        self.ceiling = ceiling
        super().__init__(*args, **kwargs)

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        ceiling, value = broadcast_all(self.ceiling, value)
        return ceiling * super().cdf(value)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        ceiling, value = broadcast_all(self.ceiling, value)
        return ceiling.log() + super().log_prob(value)

    def expand(self, *args, **kwargs):
        raise NotImplementedError


class CeilingWeibull(CeilingMixin, torch.distributions.Weibull):
    arg_constraints = {
        'scale': constraints.positive,
        'concentration': constraints.positive,
        'ceiling': constraints.unit_interval
    }


class CeilingLogNormal(CeilingMixin, torch.distributions.LogNormal):
    arg_constraints = {
        'loc': constraints.real,
        'scale': constraints.positive,
        'ceiling': constraints.unit_interval
    }
