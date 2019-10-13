import torch
from torch.distributions import ExponentialFamily
from torch.distributions import constraints

from strong_glm.distributions.ceiling import CeilingMixin


class CeilingWeibull(CeilingMixin, torch.distributions.Weibull):
    arg_constraints = {
        'scale': constraints.positive,
        'concentration': constraints.positive,
        'ceiling': constraints.unit_interval
    }
