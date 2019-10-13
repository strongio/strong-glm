import torch
from torch.distributions.utils import broadcast_all


class CeilingMixin:

    def __init__(self, ceiling: torch.Tensor, *args, **kwargs):
        self.ceiling = ceiling
        super().__init__(*args, **kwargs)

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        ceiling, value = broadcast_all(self.ceiling, value)
        return ceiling * super().cdf(value)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        ceiling, value = broadcast_all(self.ceiling, value)
        return ceiling.log() + super().log_prob(value)
