from typing import Tuple, Optional

import torch
from skorch.helper import SliceDict

from strong_glm.glm.survival.censoring import unpack_cens_y
from strong_glm.log_prob_criterion import NegLogProbLoss, _reductions


class CensNegLogProbLoss(NegLogProbLoss):
    """
    Negative-log-prob when y_true is a right-censored (and possibly left-truncated) output.
    """

    def forward(self,
                y_pred: Tuple[torch.Tensor, ...],
                y_true: torch.Tensor,
                reduction: Optional[str] = None) -> torch.Tensor:
        # y-true is a tensor with the right shape to be unpacked:
        y_vals, y_cens, y_ltrunc = unpack_cens_y(y_true)

        # y-pred is a tuple that corresponds to param_names:
        params = SliceDict(**dict(zip(self.param_names, y_pred)))

        #
        log_probs = torch.zeros_like(y_vals)
        log_probs[y_cens == 0] = self.distribution(**params[y_cens == 0]).log_prob(y_vals[y_cens == 0])
        log_probs[y_cens == 1] = self._log_surv(y_vals[y_cens == 1], params[y_cens == 1])
        if y_ltrunc is not None:
            log_probs = log_probs - self._log_surv(y_ltrunc, params)

        reduction = reduction or self.reduction
        return _reductions[reduction](-log_probs)

    def _log_surv(self, x: torch.Tensor, params_slice_dict: SliceDict) -> torch.Tensor:
        # avoid distribution at edge of PDF support, grad can be nan:
        mask = ~torch.isclose(x, torch.zeros(1))
        log_surv = torch.zeros_like(x)
        log_surv[mask] = (1 - self.distribution(**params_slice_dict[mask]).cdf(x[mask])).log()
        return log_surv
