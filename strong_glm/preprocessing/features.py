from warnings import warn

import numpy as np

from typing import Union, Sequence, Tuple

from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.utils.metaestimators import _BaseComposition

try:
    from pandas import DataFrame
except ImportError:
    def _no_pandas(*args, **kwargs):
        raise RuntimeError("Must install `pandas`")


    DataFrame = _no_pandas


def _infer_feature_names(
        input_feature_names: Sequence[str],
        output_num_cols: int,
        trans_name: str,
        transformer: TransformerMixin
) -> Sequence[str]:
    feature_names = False
    try:
        # polynomial features will add ^1, ^2, etc.
        # one_hot_encoder will add labels
        # TODO: this breaks the mapping between names in get/set_params and elsewhere feature-names. I think that's ok
        feature_names = transformer.get_feature_names(input_feature_names)
    except (TypeError, AttributeError, NotImplementedError):
        if output_num_cols == len(input_feature_names):
            if isinstance(transformer, (StandardScaler, MinMaxScaler, RobustScaler)):
                feature_names = [f"{trans_name}({fname})" for i, fname in enumerate(input_feature_names)]
            else:
                # can't assume 1-1 mapping. gotta wait for sklearn to support get_feature_names on everything
                # https://github.com/scikit-learn/scikit-learn/pull/12627
                pass

        elif len(input_feature_names) == 1:
            if output_num_cols == 1:
                feature_names = [f"{trans_name}({input_feature_names[0]})"]
            else:
                feature_names = [f"{trans_name}({input_feature_names[0]})[{i}]" for i in range(output_num_cols)]

    if feature_names is False:
        # TODO: maybe somehow support passing aliases?
        warn(f"Unable to infer feature-names for {trans_name}, forced to concatenate.")
        return _infer_feature_names(
            output_num_cols=output_num_cols,
            trans_name=trans_name,
            transformer=transformer,
            input_feature_names=input_feature_names.__repr__()
        )

    return feature_names


class FormulaArithmetic:
    def __add__(self, other):
        return FeatureList.main_effect(self, other)

    def __mul__(self, other):
        return FeatureList.main_effect(
            FeatureList.main_effect(self, other),
            FeatureList.interaction(self, other)
        )

    def __sub__(self, other):
        raise RuntimeError("TODO(@jwdink)")

    def __mod__(self, other):
        return FeatureList.interaction(self, other)


class Feature(FormulaArithmetic, TransformerMixin, _BaseComposition):
    def __init__(self,
                 feature: str,
                 transforms: Sequence[Tuple[str, TransformerMixin]] = ()):
        # TODO: make it OK to pass list of features, e.g. for PCA. maybe regexp as well?
        self.feature = feature
        self.transforms = transforms
        self.transforms_ = None
        super().__init__()

    @property
    def name(self):
        name = self.feature
        for trans_nm, trans in self.transforms:
            name = f"{trans_nm}({name})"
        return name

    def get_params(self, deep=True):
        return self._get_params('transforms', deep=deep)

    def set_params(self, **kwargs):
        self._set_params('transforms', **kwargs)
        return self

    def fit(self, X, y=None):
        self.transforms_ = []
        orig_index = X.index
        X = X.loc[:, [self.feature]]
        for trans_name, transformer in self.transforms:
            input_feature_names = list(X.columns)
            self.transforms_.append(clone(transformer))
            X = self.transforms_[-1].transform(X.values, y=y)
            X = self._standardize_transform(X, trans_name, transformer, input_feature_names, orig_index)
        return self

    def transform(self, X, y=None) -> 'DataFrame':
        orig_index = X.index
        X = X.loc[:, [self.feature]]
        for trans_name, transformer in self.transforms:
            input_feature_names = list(X.columns)
            X = transformer.transform(X.values)
            # TODO: coerce to numpy array, not df. just keep track of feature-names in a list instead of in X.columns
            X = self._standardize_transform(X, trans_name, transformer, input_feature_names, orig_index)
        return X

    @staticmethod
    def _standardize_transform(X: np.ndarray,
                               trans_name: str,
                               transformer: TransformerMixin,
                               input_feature_names: Sequence[str],
                               orig_index: 'Index'):

        if len(X.shape) == 1:
            X = X[:, None]
        return DataFrame(
            data=X,
            columns=_infer_feature_names(
                input_feature_names=input_feature_names,
                output_num_cols=X.shape[1],
                trans_name=trans_name,
                transformer=transformer
            ),
            index=orig_index
        )


# class FeatureInteraction(FormulaArithmetic, TransformerMixin, _BaseComposition):
#     # TODO: ditch this, just use tuples, move multiply logic to FeatureList
#     @classmethod
#     def flatten(cls, *args):
#         features = []
#         for arg in args:
#             if isinstance(arg, FeatureInteraction):
#                 features.extend(arg.sub_features)
#             elif isinstance(arg, Feature):
#                 features.append((arg.name, arg))
#             else:
#                 raise ValueError(f"Unexpected argument type for {arg}")
#         return cls(features)
#
#     def __init__(self, sub_features: Sequence[Tuple[str, Feature]]):
#         self.sub_features = sub_features
#         for nm, f in self.sub_features:
#             if nm != f.name:
#                 raise RuntimeError(f"sub-feature name attribute doesn't match name in argument: {nm}, {f.name}")
#         super().__init__()
#
#     @property
#     def name(self):
#         sub_feature_names, _ = zip(*self.sub_features)
#         return "%".join(sub_feature_names)
#
#     def get_params(self, deep=True):
#         if len(self.sub_features) == 1:
#             return self.sub_features[0].get_params(deep=deep)
#
#         return self._get_params('sub_features', deep=deep)
#
#     def set_params(self, **kwargs):
#         if len(self.sub_features) == 1:
#             self.sub_features[0].set_params(**kwargs)
#         else:
#             self._set_params('sub_features', **kwargs)
#         return self
#
#     def fit(self, X, y=None):
#         assert isinstance(X, pd.DataFrame)
#         for name, sub_feature in self.sub_features:
#             sub_feature.fit(X, y=y)
#         return self
#
#     def transform(self, X, y=None) -> 'DataFrame':
#         assert isinstance(X, pd.DataFrame)
#         dfs = []
#         for name, sub_feature in self.sub_features:
#             dfs.append(sub_feature.transform(X))
#         if len(dfs) == 1:
#             return dfs[0]
#         if len(dfs) > 2:
#             raise RuntimeError("TODO(@jwdink)")
#         else:
#             out = pd.DataFrame(index=X.index)
#             for col1 in dfs[0].columns:
#                 for col2 in dfs[1].columns:
#                     out[f"{col1}%{col2}"] = dfs[0][col1] * dfs[1][col2]
#
#         return out


class FeatureList(FormulaArithmetic, BaseEstimator, TransformerMixin):
    @classmethod
    def main_effect(cls, lhs: Union[Feature, 'FeatureList'], rhs: Union[Feature, 'FeatureList']) -> 'FeatureList':
        lhs, rhs = cls._std_args(lhs, rhs)
        raise NotImplementedError("TODO")

        final_features = list(lhs.features)
        final_feature_names = set(nm for nm, _ in final_features)
        for feature_name, feature in rhs.features:
            if feature_name not in final_feature_names:
                final_features.append((feature_name, feature))

        return cls(final_features)

    @classmethod
    def interaction(cls, lhs: Union[Feature, 'FeatureList'], rhs: Union[Feature, 'FeatureList']):
        lhs, rhs = cls._std_args(lhs, rhs)
        raise NotImplementedError("TODO")

        interaction_features = []
        fsets = []
        for fname1, ffeat1 in lhs.features:
            for fname2, ffeat2 in rhs.features:
                fset = {fname1, fname2}
                if (fname1 == fname2) or (fset in fsets):
                    continue
                fsets.append(fset)
                interaction = FeatureInteraction([(fname1, ffeat1), (fname2, ffeat2)])
                interaction_features.append((interaction.name, interaction))
        return cls(interaction_features)

    def __init__(self, groups: Sequence[str], features: Sequence[FeatureInteraction]):
        self.features = features
        self.features_ = None
        super().__init__()

    def fit(self, X, y=None):
        raise NotImplementedError("TODO")

    def transform(self, X, y=None) -> 'DataFrame':
        # TODO: handle FeatureInteractions here
        # dfs = [feature.transform(X).reset_index(drop=True) for nm, feature in self.features]
        raise NotImplementedError("TODO")

    def get_params(self, deep=True):
        # see _BaseComposition._get_params()
        raise NotImplementedError("TODO")

    def set_params(self, **kwargs):
        # _BaseComposition._set_params()
        raise NotImplementedError("TODO")

    def __repr__(self):
        # TODO: I don't like this
        names, _ = zip(*self.features)
        return "FeatureList(\n~{}\n)".format(" + ".join(names))

    @staticmethod
    def _std_args(lhs: Union[Feature, 'FeatureList'], rhs: Union[Feature, 'FeatureList']):
        # TODO: tuples instead of FeatureInteractions
        raise NotImplementedError("TODO")
        if isinstance(lhs, Feature):
            lhs = FeatureInteraction([(lhs.name, lhs)])
        if isinstance(lhs, FeatureInteraction):
            lhs = FeatureList([(lhs.name, lhs)])
        if isinstance(rhs, Feature):
            rhs = FeatureInteraction([(rhs.name, rhs)])
        if isinstance(rhs, FeatureInteraction):
            rhs = FeatureList([(rhs.name, rhs)])
        return lhs, rhs


def feature(nm: str, *args) -> Feature:
    transforms = []
    for trans in args:
        if not isinstance(trans, BaseEstimator) and callable(trans):
            name = trans.__name__.lower()
            trans = FunctionTransformer(trans, validate=False)
        else:
            name = trans.__class__.__name__.lower()
        transforms.append((name, trans))
    return Feature(nm, transforms)


def features(nms: Sequence[str], *args) -> FeatureList:
    out = None
    for nm in nms:
        feat = feature(nm, *[clone(arg, safe=False) for arg in args])
        if out is None:
            raise NotImplementedError("TODO")
            # out = FeatureList([(feat.name, FeatureInteraction(sub_features=[(feat.name, feat)]))])
        else:
            raise NotImplementedError("TODO")
            # out = FeatureList.main_effect(out, feat)
    return out


