import torch

from strong_glm.penalty.base import Penalty


class NoPenalty(Penalty):
    def __init__(self, multi: float = 0.0):
        super().__init__(multi=multi)

    @staticmethod
    def calculate_penalty(input, target):
        return torch.zeros(1)


class SmoothL1(Penalty):
    calculate_penalty = torch.nn.modules.loss.SmoothL1Loss(reduction='sum')


class L2(Penalty):
    calculate_penalty = torch.nn.modules.loss.MSELoss(reduction='sum')
