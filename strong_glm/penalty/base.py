import torch


class Penalty(torch.nn.Module):
    calculate_penalty = None

    def __init__(self, multi: float):
        assert isinstance(multi, float)
        assert multi >= 0.0
        self.multi = multi
        super().__init__()

    @staticmethod
    def is_bias(param_name: str) -> bool:
        return ('bias' in param_name.lower()) or ('intercept' in param_name.lower())

    def forward(self, **kwargs) -> torch.Tensor:
        penalties = torch.zeros(len(kwargs))
        for i, (param_name, param) in enumerate(kwargs.items()):
            if not self.is_bias(param_name):
                penalties[i] = self.calculate_penalty(param, torch.zeros_like(param))
        return self.multi * torch.sum(penalties)
