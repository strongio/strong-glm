# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# This notebook can be opened in jupyter notebook with https://github.com/mwouts/jupytext

# +
# %matplotlib inline

import torch
import numpy as np
import pandas as pd

from plotnine import *
from mizani import formatters

torch.manual_seed(2019-10-26)
np.random.seed(2019-10-26)
# -

# ## Simulate Data

# +
from strong_glm.utils import simulate_data

true_betas = np.random.laplace(size=40) / 10
data = simulate_data(num_rows=2000, linear_pred_betas=true_betas[:20], binary_pred_betas=true_betas[20:])
data['y'] = torch.distributions.Poisson(rate=torch.from_numpy(np.exp(data['y'].values))).sample().numpy()
data.head()
# -

print(
    ggplot(data, aes(x='y')) + 
    stat_count() + theme_bw() + 
    geom_hline(yintercept=0) +
    scale_x_continuous(breaks=range(0,16)) +
    theme(figure_size=(8,4))
)

# ## Fit Linear Model with L2-Penalty (Ridge) using Sklearn

# ### Set up Data-Preprocessing

# +
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer

input_feature_names = data.columns[data.columns.str.startswith('x')].tolist()

preproc = make_column_transformer(
    (StandardScaler(), input_feature_names)
)
# -

# ### Grid-Search CV

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

# +
from sklearn.linear_model import Ridge

sklearn_ridge = Ridge(fit_intercept=True, normalize=False)
sklearn_ridge_grid = GridSearchCV(
    estimator=make_pipeline(preproc, sklearn_ridge),
    param_grid={'ridge__alpha' : 10 ** np.linspace(-1,2.5,20)},
    cv=5,
    scoring='neg_mean_squared_error'
)

# fit:
sklearn_ridge_grid.fit(X=data, y=data['y'].values)

# get cv results:
sklearn_ridge_grid.cv_df_ = pd.DataFrame(sklearn_ridge_grid.cv_results_)
sklearn_ridge_grid.cv_df_['penalty'] =\
    sklearn_ridge_grid.cv_df_.pop('params').apply(lambda p: p ['ridge__alpha'])

print(
    ggplot(sklearn_ridge_grid.cv_df_, 
           aes(x='penalty', y='-mean_test_score')) +
    geom_line() +
    geom_point(data=sklearn_ridge_grid.cv_df_.query("rank_test_score==1")) +
    scale_x_continuous(name="Penalty", trans='log10') +
    scale_y_continuous(name="Avg. Validation MSE (5-Folds)") +
    ggtitle("Sklearn Ridge: Cross-Validation Performance") +
    theme_bw()
)
# -

# ## Fit Poisson Model with L2-Penalty using Strong-Glm

# +
from strong_glm.glm import Glm

strong_ridge = Glm(distribution=torch.distributions.Poisson)
strong_ridge_grid = GridSearchCV(
    estimator=make_pipeline(preproc, strong_ridge),
    param_grid={'glm__criterion__penalty' : 10 ** np.linspace(-1,2.5,20)},
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

# fit:
strong_ridge_grid.fit(X=data, y=data['y'].values, glm__input_feature_names=input_feature_names)

# get cv results:
strong_ridge_grid.cv_df_ = pd.DataFrame(strong_ridge_grid.cv_results_)
strong_ridge_grid.cv_df_['penalty'] =\
    strong_ridge_grid.cv_df_.pop('params').apply(lambda p: p ['glm__criterion__penalty'])
# -

# ## Compare Results

# +
df_perf_compare = pd.concat([
    strong_ridge_grid.cv_df_.loc[:,['penalty', 'mean_test_score','rank_test_score']].assign(model='strong'), 
    sklearn_ridge_grid.cv_df_.loc[:,['penalty', 'mean_test_score','rank_test_score']].assign(model='sklearn')
],
    ignore_index=True
)

print(
    ggplot(df_perf_compare, 
           aes(x='penalty', y='-mean_test_score', color='model')) +
    geom_line() +
    geom_point(data=df_perf_compare.query("rank_test_score==1")) +
    scale_x_continuous(name="Penalty", trans='log10') +
    scale_y_continuous(name="Avg. Validation MSE (5-Folds)") +
    ggtitle("Performance Comparison") +
    theme_bw() +
    theme(figure_size=(8,4))
)
# -
# ## Visualize Coefficient-Estimates

# +
preproc = strong_ridge_grid.best_estimator_[0]
glm = strong_ridge_grid.best_estimator_._final_estimator
glm.estimate_laplace_params(X=preproc.transform(data), y=data['y'].values)
df_params = glm.summarize_laplace_params()
df_params['ground_truth'] = df_params['feature'].map(dict(zip(input_feature_names, true_betas)))

print(
    ggplot(df_params, aes(x='estimate', y='ground_truth')) +
    geom_point() +
    geom_errorbarh(aes(xmin='estimate - std', xmax='estimate + std'), alpha=.50, height=0.0001) +
    geom_abline() +
    theme(figure_size=(10,10)) +
    geom_hline(yintercept=0, linetype='dashed') +
    theme_bw() +
    scale_x_continuous(name="Model-Estimate") + scale_y_continuous(name="Ground-Truth")
)
# -


