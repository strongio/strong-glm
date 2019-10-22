import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from typing import Sequence, Optional, Union, Tuple


class Segmented:
    def __init__(self,
                 breaks: Union[int, Sequence] = (),
                 break_method: Optional[str] = None):
        self.breaks = breaks
        self.break_method = break_method
        self.break_locations_ = None

    def _choose_breaks(self,
                       X: np.ndarray,
                       y: Optional[np.ndarray],
                       num: int) -> Tuple[float, ...]:
        break_method = (self.break_method or 'quantile').lower()

        if break_method == 'quantile':
            q = np.linspace(0, 1, num + 2)[1:-1]
            breaks = set(np.nanquantile(X, q))
            if len(breaks) != num:
                raise RuntimeError("quantile method returned duplicate breaks")
            return tuple(sorted(breaks))
        else:
            raise NotImplementedError

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        X = self._validate_X(X)
        assert len(np.unique(X)) > 1

        if isinstance(self.breaks, (list, tuple, np.ndarray)):
            assert len(self.breaks) == len(set(self.breaks))
            self.break_locations_ = tuple(self.breaks)
        else:
            self.break_locations_ = self._choose_breaks(X, y, self.breaks)

        return self

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        if hasattr(X, 'values'):
            X = X.values
        if len(X.shape) == 2:
            if X.shape[1] != 1:
                raise RuntimeError("Expected X to be vector")
            X = X.squeeze(-1)
        return X


class SegmentedBoundary(Segmented):
    def __init__(self,
                 breaks: Union[int, Sequence] = (),
                 boundary_breaks: Optional[Sequence] = None,
                 break_method: Optional[str] = None):
        self.boundary_breaks = boundary_breaks
        super().__init__(breaks=breaks, break_method=break_method)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        super().fit(X, y)
        s, e = self._choose_boundary(X)

        if not all(s < k < e for k in self.break_locations_):
            raise ValueError(f"Some breaks fall outside boundary: {s, e}")

        self.break_locations_ = (s,) + self.break_locations_ + (e,)

        return self

    def _choose_boundary(self, X: np.ndarray) -> Tuple:
        if isinstance(self.boundary_breaks, (list, tuple, np.ndarray)):
            return self.boundary_breaks
        else:
            return np.nanmin(X), np.nanmax(X)


class PiecewiseCumulative(Segmented, BaseEstimator, TransformerMixin):
    def transform(self, X: np.ndarray, y=None):
        X = self._validate_X(X)
        out = [X]
        raise RuntimeError("TODO: why [1:]?")
        for brk in self.break_locations_[1:]:
            out.append(np.maximum(X - brk, 0))

        out = np.vstack(out).T
        return out


class Piecewise(SegmentedBoundary, BaseEstimator, TransformerMixin):
    def __init__(self,
                 breaks: Union[int, Sequence] = (),
                 boundary_breaks: Optional[Sequence] = None,
                 break_method: Optional[str] = None,
                 dampen: float = 0.0):
        """

        :param breaks: Either (a) integer for number of breaks to find, if break_method supports this, (b) actual
        breakpoints. The piecewise function has num-breaks + 1 degrees of freedom.
        :param break_method: Method for finding breaks; default is to use quantiles. Ignored if actual breakpoints are
        specified w/`breaks` argument.
        :param boundary_breaks: In addition to the breaks chosen with `breaks`, these outer breaks will be added. The
        function's behavior outside of these breaks is determined by the `dampen` argument. TODO: If dampen is 0.0,
        placement of boundary breaks has no effect? TODO: default behavior is minmax is dampen is zero, else (TODO)
        :param dampen: A float between 0 and 1 inclusive. If 1.0, then the function is flat outside of the boundary-
        breaks. If 0.0, then the function extrapolates the slope from before the boundary.
        :param control_break_idx: TODO
        """
        assert 0.0 <= dampen <= 1.0
        self.dampen = dampen
        super().__init__(breaks=breaks, break_method=break_method, boundary_breaks=boundary_breaks)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        super().fit(X, y)

    def transform(self, X: np.ndarray, y=None):
        X = self._validate_X(X)

        num_knots = len(self.break_locations_)
        if num_knots == 2:
            return X

        spans = np.diff(self.break_locations_)
        spans /= spans.mean()

        out = []

        # LHS:
        for i in range(1, middle_idx + 1):
            clipped = np.clip(X, None, self.break_locations_[i])
            x = self._dampen(clipped, 'lhs') - self.break_locations_[i]
            if self.scale_by_span:
                x *= spans[i - 1]
            out.append(x)

        # RHS:
        for i in range(middle_idx, num_knots - 1):
            clipped = np.clip(X, self.break_locations_[i], None)
            x = self._dampen(clipped, 'rhs') - self.break_locations_[i]
            if self.scale_by_span:
                x *= spans[i]
            out.append(x)

        return np.vstack(out).T

    def _dampen(self, clipped: np.ndarray, side: str) -> np.ndarray:
        # TODO: could make dampening increase with distance?
        x = clipped.copy()
        w1 = self.dampen
        w2 = 1 - w1
        if side == 'rhs':
            outer_break = self.break_locations_[-1]
            is_outer = x > outer_break
        else:
            outer_break = self.break_locations_[0]
            is_outer = x < outer_break
        x[is_outer] = (w1 * outer_break + w2 * x)[is_outer]
        return x

    def _choose_boundary(self, X: np.ndarray) -> Tuple:
        s, e = super()._choose_boundary(X)
        if not len(self.break_locations_) or self.dampen == 0.0:
            return s, e
        raise RuntimeError("TODO: mean of (last_quantile,100.0), etc.")
        return s, e



try:
    from patsy.mgcv_cubic_splines import _get_free_crs_dmatrix, _get_centering_constraint_from_dmatrix, _get_crs_dmatrix
except ImportError:
    def _no_patsy(*args, **kwargs):
        raise ImportError("Must install `patsy` for cardinal basis")


    _get_free_crs_dmatrix = _get_centering_constraint_from_dmatrix = _get_crs_dmatrix = _no_patsy


class NaturalSpline(SegmentedBoundary, BaseEstimator, TransformerMixin):
    def __init__(self,
                 knots: Union[int, Sequence] = (),
                 boundary_knots: Optional[Sequence] = None,
                 knot_select_method: Optional[str] = None,
                 basis: str = 'cardinal'):
        self.basis = basis
        self.constraints_ = None
        super().__init__(breaks=knots, boundary_breaks=boundary_knots, break_method=knot_select_method)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        super().fit(X, y)
        if self.basis.lower() == 'cardinal':
            self.constraints_ = _get_centering_constraint_from_dmatrix(_get_free_crs_dmatrix(x, self.break_locations_))
        return self

    def transform(self, X: np.ndarray, y=None):
        X = self._validate_X(X)
        if self.basis.lower().startswith('trunc'):
            return self._trunc_power_basis(X)
        elif self.basis.lower() == 'cardinal':
            return self._cardinal_basis(X)
        else:
            raise ValueError(f"Unrecognized basis: {self.basis}")

    def _cardinal_basis(self, X) -> np.ndarray:
        return _get_crs_dmatrix(X, knots=self.break_locations_, constraints=self.constraints_)

    def _trunc_power_basis(self, X: np.ndarray) -> Sequence:
        out = [X]
        sub = self._d(X, self.break_locations_[-2])
        for knot in self.break_locations_[:-2]:
            out.append(self._d(X, knot) - sub)
        return np.vstack(out).T

    def _d(self, X: np.ndarray, knot: float) -> np.ndarray:
        last_knot = self.break_locations_[-1]
        p1 = np.maximum(X - knot, 0) ** 3
        p2 = np.maximum(X - last_knot, 0) ** 3
        return (p1 - p2) / (last_knot - knot)
