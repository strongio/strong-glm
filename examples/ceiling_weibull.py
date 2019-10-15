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

torch.manual_seed(2019-10-11)
np.random.seed(2019-10-11)
# -

# ## Simulate Data

# +
N = 2000
PROP_NEVER_CONVERT = .30
true_intercept = 4.5
true_betas = np.array([0.5, 0., -0.5])

data = pd.DataFrame({
     'start_time' : np.round(np.random.beta(1.5,1,N) * 365.),
     'X1' : np.random.randn(N),
     'X2' : np.random.randn(N) ** 2
})
data['customer_id'] = data.index

X = data.loc[:,['start_time','X1','X2']]
X /= X.mean()

true_weibull = torch.distributions.Weibull(
    scale=torch.Tensor(np.exp(true_intercept + X.values @ true_betas)),
    concentration=1.2
)

# % of customers will never convert, no matter what:
data['_tte_true'] = np.where(np.random.rand(N) > PROP_NEVER_CONVERT,
                           true_weibull.rsample().numpy().round(),
                           np.inf)

data['tte'] = data['_tte_true'].where(data['start_time'] + data['_tte_true'] < 365., other=365. - data['start_time'])
data['tte'] += 1.
data['censored'] = data['_tte_true'] > data['tte']
data.head()
# -

print(
    ggplot(data.query("customer_id.isin(customer_id.sample(100))"), aes(color='censored')) + 
      geom_segment(aes(x='start_time', xend='start_time+tte', 
                       y='factor(customer_id)', yend='factor(customer_id)')) +
      coord_cartesian(xlim=(0,365)) +
    scale_x_continuous(name='') + scale_y_discrete(name="Customer", breaks=None) +
    theme_bw() +
    scale_color_manual(values=("blue", "gray")) +
    geom_vline(xintercept=365, linetype='dashed') +
    theme(legend_position='none', figure_size=(8,4))
)

# ## Survival Model

# ### Fit

# +
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer

input_feature_names = ['X1','X2','start_time']

preproc = make_column_transformer(
    (StandardScaler(), input_feature_names)
)
preproc.fit(data)

# +
from strong_glm.glm.survival import SurvivalGlm
from torch.distributions import Weibull

model_no_ceiling = SurvivalGlm(
    distribution=Weibull, 
    lr=.10,
    scale_y=True
)
model_no_ceiling.fit(X=preproc.transform(data), 
                     y=data.loc[:,['tte','censored']].values,
                     input_feature_names=input_feature_names)

# +
from strong_glm.distributions import CeilingWeibull

model_ceiling = SurvivalGlm(
    distribution=CeilingWeibull, 
    lr=.10,
    scale_y=True
)
model_ceiling.fit(X=preproc.transform(data), 
                  y=data.loc[:,['tte','censored']].values,
                  input_feature_names=input_feature_names)
# -

# ### Diagnostics/Viz

# #### Compare Choice of Distribution
#
# Model with a 'ceiling' parameter appears to be a better fit.

# +
df_km = pd.concat([
    model_ceiling.km_summary(data, preproc, time_colname='tte', censor_colname='censored').assign(type='Weibull w/Ceiling'),
    model_no_ceiling.km_summary(data, preproc, time_colname='tte', censor_colname='censored').assign(type='Weibull')
])

print(
    ggplot(df_km, aes(x='time')) + 
    geom_step(aes(y='km_estimate')) + 
    geom_line(aes(y='model_estimate', color='type'), alpha=.50, size=2) +
    geom_hline(yintercept=0) +
    theme_bw() +
    scale_color_brewer(name="Probability Distribution", type="qual", palette="Set1") +
    scale_y_continuous(name="KM Estimate (black) vs. Model-Estimates", labels=formatters.percent) +
    scale_x_continuous(name="Time") +
    theme(legend_position=(.7,.7), figure_size=(6,5))
)

# +
from strong_glm.glm.survival.censoring import cens_y_to_indicator

data['pred_60'] = model_ceiling.predict(preproc.transform(data), type='cdf', value=60/model_ceiling.y_scaler_.mean_.item())
data['actual_60'] = cens_y_to_indicator(time=data['tte'], is_upper_cens=data['censored'], window=60)

print(
    ggplot(data, aes(x='pred_60', y='actual_60')) + 
    stat_summary_bin(fun_data='mean_cl_boot') +
    geom_hline(yintercept=(0,1)) +
    theme_bw() + geom_abline(linetype='dashed') +
    scale_x_continuous(name="Actual (horizon=60)", labels=formatters.percent, limits=(0,1)) +
    scale_y_continuous(name="Predicted (horizon=60)", labels=formatters.percent) 
)
# -

# #### Visualize Predictors

# +
df_km = model_ceiling.km_summary(
    dataframe=data.groupby(pd.qcut(np.round(data['X2'],2), 3)),
    preprocessor=preproc,
    time_colname='tte', 
    censor_colname='censored'
)

print(
    ggplot(df_km, aes(x='time', color='X2')) + 
    geom_step(aes(y='km_estimate')) + 
    geom_line(aes(y='model_estimate'), alpha=.50, size=2) +
    geom_hline(yintercept=0) +
    theme_bw() +
    scale_color_brewer(name="X2 Predictor (Binned)", type="qual", palette="Set1") +
    scale_y_continuous(name="KM vs. Model Estimates", labels=formatters.percent) +
    scale_x_continuous(name="Time") +
    theme(legend_position=(.7,.7), figure_size=(6,5))
)

