import logging

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

import statsmodels.tsa.api as smt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklego.pandas_utils import log_step

logging.basicConfig(level=logging.INFO)

@log_step
def remove_inconsistencies(df):
    "Replace outliers values (999.90) with median value for that month"
    
    for col in df.columns:
        median_col = df[col].median()
        df[col] = df[col].replace(999.90, median_col)
        
    return df

@log_step
def get_only_month_columns(df):
    return df.iloc[:, :13]

def preprocess_climate_data(df_climate):
    return (df_climate
                   .rename(columns=str.lower)
                   .pipe(get_only_month_columns)
                   .pipe(remove_inconsistencies)
                   .melt(id_vars='year',
                         var_name='month',
                         value_name='temperature')
                   .assign(date=lambda x: pd.to_datetime(x['year'].astype(str) + '-' + x['month']))
                   .drop(['year', 'month'], axis=1)
                   .set_index('date')
                   .sort_index()
                   .copy()
            )

def collect_results(pdmarima_models, test):
    maes, rmses, aics, bics, orders = list(), list(), list(), list(), list()
    
    for pdmarima_model in pdmarima_models:
        forecast = pdmarima_model.predict(test.shape[0])

        maes.append(np.round(mean_absolute_error(test.values, forecast), 2))
        rmses.append(np.round(np.sqrt(mean_squared_error(test.values, forecast)), 2))
        aics.append(np.round(pdmarima_model.aic(), 2))
        bics.append(np.round(pdmarima_model.bic(), 2))
        orders.append(str(pdmarima_model.order))

    fig = go.Figure(data=[go.Table(header=dict(values=['Model Order', 'AIC', 'BIC', 'RMSE', 'MAE']),
                    cells=dict(values=[orders, aics, bics, rmses, maes]))
                         ])
    fig.update_layout(width=550, height=400)
    
    return fig




def tsplot(y, lags=None, figsize=(10, 8), **kargs):
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))

    y.plot(ax=ts_ax)
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, **kargs)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    sns.despine()
    plt.tight_layout()

    return ts_ax, acf_ax, pacf_ax