import unittest

import torch

from strong_glm.distributions.piecewise import Piecewise
from scipy import integrate


class TestPiecewise(unittest.TestCase):
    def test_cumu_haz(self):
        pwise = Piecewise(knots=(2, 6), k0=-3, k1=-.5, k2=.5)

        def _cumu_haz_numerical(x, haz):
            return [integrate.quad(haz, 0, xi, limit=100)[0] for xi in x]

        x = torch.arange(0, 10, .2)
        df = {
            'haz': pwise.log_hazard(x).exp().numpy(),
            'cumu_haz': _cumu_haz_numerical(x, lambda x: pwise.log_hazard(x).exp()),
            'cumu_haz2': pwise.cumu_hazard(x).numpy(),
            'cdf': pwise.cdf(x).numpy(),
            'pdf': pwise.log_prob(x).exp().numpy()
        }
        self.assertLess(max(abs(df['cumu_haz'] - df['cumu_haz2'])), .001)
