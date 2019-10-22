from collections import OrderedDict
from typing import Sequence, Optional, Tuple, Type

from numbers import Number

import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all

from strong_glm.distributions.ceiling import CeilingMixin


class Piecewise(torch.distributions.ExponentialFamily):
    support = constraints.positive
    arg_constraints = None
    _num_knots = None

    def __init_subclass__(cls, **kwargs):
        if cls._num_knots is not None:
            # if subclass defines the _num_knots, then constraints can be a class-attribute (otherwise only
            # instance-attr, defined in __init__ below).
            cls.arg_constraints = {"k{}".format(i): constraints.real for i in range(cls._num_knots + 1)}
        return super().__init_subclass__()

    def __init__(self,
                 knots: Sequence[float] = None,
                 validate_args: Optional[bool] = None,
                 **kwargs):
        if len(knots) != len(set(knots)):
            raise ValueError("Knots are not unique")
        if self._num_knots is not None:
            if len(knots) != self._num_knots:
                raise TypeError("`{}` expects {} knots.".format(type(self).__name__, self._num_knots))
        self.knots = sorted(float(k) for k in knots)

        param_names = []
        params = []
        for i in range(self.num_knots + 1):
            k = "k{}".format(i)
            if k not in kwargs:
                raise ValueError("Missing argument for param {}.".format(k))
            param_names.append(k)
            params.append(kwargs[k])
        params = broadcast_all(*params)
        self.params = OrderedDict(zip(param_names, params))
        if not self.arg_constraints:
            self.arg_constraints = {k: constraints.real for k in param_names}
        batch_shape = torch.Size() if isinstance(self.params['k0'], Number) else self.params['k0'].size()
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def num_knots(self):
        return len(self.knots)

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        return 1 - torch.exp(-self.cumu_hazard(value))

    def log_prob(self, value: torch.Tensor):
        """
        hazard = prob / surv
        surv = exp(-cumu_hazard)
        log(hazard) = log(prob) - log(exp(-cumu_hazard))
        log(hazard) = log(prob) + cumu_hazard
        log(prob) = log(hazard) - cumu_hazard
        """
        return self.log_hazard(value) - self.cumu_hazard(value)

    def log_hazard(self, value: torch.Tensor) -> torch.Tensor:
        value, *param_values = broadcast_all(value, *self.params.values())
        log_hazard = [param_values[0]]
        for knot, param in zip(self.knots, param_values[1:]):
            log_hazard.append(param * torch.clamp(knot - value, 0.0))
        return torch.sum(torch.stack(log_hazard), 0)

    def cumu_hazard(self, value: torch.Tensor) -> torch.Tensor:
        times, *param_values = broadcast_all(value, *self.params.values())

        knots_plus = [0.0] + self.knots + [float('inf')]

        out = []
        for knot_idx in range(1, self.num_knots + 2):
            knot_times = torch.clamp(times, max=knots_plus[knot_idx])
            params_and_knots = list(zip(param_values[knot_idx:], knots_plus[knot_idx:]))
            b = self._indef_integral(
                times=knot_times,
                params_and_knots=params_and_knots,
                intercept=param_values[0]
            )
            a = self._indef_integral(
                times=torch.full_like(times, knots_plus[knot_idx - 1]),
                params_and_knots=params_and_knots,
                intercept=param_values[0]
            )

            out.append(torch.clamp(b - a, 0.0))

        return torch.sum(torch.stack(out), 0)

    def _indef_integral(self,
                        times: torch.Tensor,
                        params_and_knots: Sequence[Tuple],
                        intercept: torch.Tensor) -> torch.Tensor:
        if not times.numel():
            return times
        if params_and_knots:
            params, _ = zip(*params_and_knots)
            denom = torch.sum(torch.stack(params), 0)
            is_near_zero = torch.isclose(denom, torch.zeros(1))
            nz_adj = torch.sum(torch.stack([p * k for p, k in params_and_knots]), 0)
        else:
            is_near_zero = torch.ones_like(intercept).to(torch.bool)
            nz_adj = torch.zeros_like(intercept)

        if not is_near_zero.any():
            log_numer = [intercept]
            for param, knot in params_and_knots:
                log_numer.append(param * (knot - times))
            log_numer = torch.sum(torch.stack(log_numer), 0)
            return - torch.exp(log_numer) / denom

        out = torch.zeros_like(intercept)
        out[is_near_zero] = torch.exp(intercept[is_near_zero] + nz_adj[is_near_zero]) * times[is_near_zero]
        out[~is_near_zero] = self._indef_integral(
            times=times[~is_near_zero],
            params_and_knots=[(p[~is_near_zero], k) for p, k in params_and_knots],
            intercept=intercept[~is_near_zero]
        )
        return out

    def expand(self, batch_shape: torch.Size, _instance: Optional[torch.distributions.Distribution] = None):
        if _instance is not None:
            raise NotImplementedError
        batch_shape = torch.Size(batch_shape)
        new = type(self)(knots=self.knots, **{k: v.expand(batch_shape) for k, v in self.params.items()})
        return new

    @property
    def mean(self):
        raise NotImplementedError

    @property
    def variance(self):
        raise NotImplementedError

    def rsample(self, sample_shape: torch.Size = torch.Size()):
        raise NotImplementedError

    def icdf(self, value):
        raise NotImplementedError

    def enumerate_support(self, expand=True):
        raise NotImplementedError


class CeilingPiecewise(CeilingMixin, Piecewise):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.arg_constraints:
            cls.arg_constraints.update(ceiling=constraints.unit_interval)


def piecewise_distribution(num_knots: int, ceiling: bool = False) -> Type[Piecewise]:
    if ceiling:
        return type("CeilingPiecewise{}".format(num_knots), (CeilingPiecewise,), {'_num_knots': num_knots})
    else:
        return type("Piecewise{}".format(num_knots), (Piecewise,), {'_num_knots': num_knots})
