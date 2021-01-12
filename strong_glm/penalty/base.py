from typing import Union, Dict

import torch

from strong_glm.glm.utils import MultiOutputModule


class Penalty(torch.nn.Module):
    """
    Penalizes a nn.Module's parameters, except for the bias parameters.
    """
    calculate_penalty = None

    def __init__(self, multi: Union[float, Dict]):
        assert isinstance(multi, (float, dict))
        self.multi = multi
        super().__init__()

    def forward(self, module: MultiOutputModule) -> torch.Tensor:
        penalties = torch.zeros(len(module))
        for i, (dist_param, sub_module) in enumerate(module.items()):
            for nm, params in sub_module.named_parameters():
                if nm == 'bias':
                    continue
                if not params.numel():
                    continue
                assert nm == 'weight'
                multi = self.multi[dist_param] if isinstance(self.multi, dict) else self.multi
                assert multi >= 0.0
                penalties[i] = multi * self.calculate_penalty(params, torch.zeros_like(params))
        return torch.sum(penalties)
