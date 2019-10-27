from typing import Optional

import torch

from typing.re import Pattern


class Penalty(torch.nn.Module):
    """
    Penalizes a nn.Module's parameters, except for the bias parameters.
    """
    calculate_penalty = None
    bias_name_pattern: Optional[Pattern] = None

    def __init__(self, multi: float):
        assert isinstance(multi, float)
        assert multi >= 0.0
        self.multi = multi
        super().__init__()

    def is_bias(self, param_name: str) -> bool:
        if self.bias_name_pattern is None:
            return ('bias' in param_name.lower()) or ('intercept' in param_name.lower())
        else:
            return self.bias_name_pattern.match(param_name)

    def forward(self, **named_params) -> torch.Tensor:
        penalties = torch.zeros(len(named_params))
        for i, (param_name, param) in enumerate(named_params.items()):
            if not self.is_bias(param_name):
                penalties[i] = self.calculate_penalty(param, torch.zeros_like(param))
        return self.multi * torch.sum(penalties)
