import torch

from strong_glm.penalty.base import Penalty


class SmoothL1(Penalty):
    calculate_penalty = torch.nn.modules.loss.SmoothL1Loss(reduction='sum')


class L2(Penalty):
    calculate_penalty = torch.nn.modules.loss.MSELoss(reduction='sum')
