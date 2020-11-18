#!/usr/bin/env python3
"""Predicts Crypto Returns using Random Forest & Sentiment-based Features.

Use 2 Random Forest models to predict returns using provided features.
Finally, compare 3 Models (RF Classifier, RF Regressor, VAR) in Cryptocurrency
Returns Predictability.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# 1. Data Preparation
data = pd.read_csv('resources/input/final_dataset.csv', index_col='Date')

# Market Returns - Dependent Variable
y = data['CRIX'].diff() / data['CRIX'].shift(1)
y.rename('ret', inplace=True)

# Independent Variables
X = data.iloc[:, 1:]

# 2. Random Forest Regressor
list_lags = []
for i in range(1, 6):
    X_lag = X.shift(i)
    for name in X.columns:
        X_lag.rename(columns={name: f"{name}({-i})"}, inplace=True)
    list_lags.append(X_lag)
    del X_lag

# Features - lagged values of sentiment measures
X_lagged = pd.concat(list_lags, axis=1)
X_lagged.dropna(inplace=True)
y = y.loc[X_lagged.index[0]:]

# split train & set data
X_train, X_test, y_train, y_test = train_test_split(X_lagged, y, shuffle=False)
y_test_num = y_test  # for later evaluation of VAR model (calculate MSFE)

RANDOM_STATE = 42
rfr = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE,
                            n_jobs=-1, max_features='sqrt')
rfr.fit(X_train, y_train)
y_fitted_rfr = rfr.predict(X_test)

# calculate MSFE of RF Regressor
error_rfr = y_fitted_rfr - y_test
mse_rfr = sum(error_rfr * error_rfr) / len(error_rfr)
print(f'MSFE Random Forest Regressor: {mse_rfr}')


def sign_ret(num):
    """
    Convert predicted returns into trading signals
    Parameters
    ----------
    num : int
        Any number (in our case predicted returns)

    Returns
    -------
    int
        Trading signals, depends on the signs of predicted returns
        (1 is go long, -1 is go short, 0 is stay still)

    """
    if num > 0:
        return 1
    elif num < 0:
        return -1
    elif num == 0:
        return 0


y_fitted_rfr = pd.Series(y_fitted_rfr).apply(sign_ret)
y_test = y_test.apply(sign_ret)

y_train_fit_rfr = pd.Series(rfr.predict(X_train)).apply(sign_ret)
y_train = y_train.apply(sign_ret)


# Print confusion matrices of train, test data
def evaluate(y_test_fit, y_test_real, y_train_fit=None, y_train_real=None):
    """
    Print confusion matrices and predictive accuracy
    (to evaluate a model's performance).
    Parameters
    ----------
    y_test_fit : Series
        y predicted using test data
    y_test_real : Series
        real y during the test period
    y_train_fit : Series
        y predicted using train data
    y_train_real : Series
        real y during the training period

    Returns
    -------
    NoneType
        None
    """
    test_cm = confusion_matrix(y_test_real, y_test_fit)
    cm_index = ['Negative', 'Zero', 'Positive']
    d_test_cm = pd.DataFrame(test_cm, columns=cm_index, index=cm_index)

    accuracy = ((d_test_cm.iloc[0][0] + d_test_cm.iloc[1][1] +
                 d_test_cm.iloc[2][2]) / d_test_cm.sum().sum())

    print('Model Performance')
    print(f'Prediction Accuracy: {round(accuracy * 100, 2)}%.')
    print('--------')
    print('Confusion Matrix (Test Data):')

    print(d_test_cm)
    print('--------')
    if isinstance(y_train_fit, pd.Series):
        train_cm = confusion_matrix(y_train_real, y_train_fit)
        d_train_cm = pd.DataFrame(train_cm, columns=cm_index, index=cm_index)
        print('Confusion Matrix (Train Data):')
        print(d_train_cm)
    print('--------')


evaluate(y_fitted_rfr, y_test, y_train_fit_rfr, y_train)

# Random Forest Classifier

rfc = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE,
                             n_jobs=-1, max_features='sqrt')

rfc.fit(X_train, y_train)
y_fitted_rfc = pd.Series(rfc.predict(X_test))

evaluate(y_fitted_rfc, y_test, rfc.predict(X_train), y_train)

# Trading Strategies
y_fitted_rfr.index = y_fitted_rfc.index = y_test.index
ret = y.loc[y_test.index[0]:]
strats = pd.DataFrame()

# Compute cumulative returns of strategies
strats['Buy-and-Hold'] = (1 + ret).cumprod()
strats['RF Regressor'] = (1 + ret * y_fitted_rfr).cumprod()
strats['RF Classifier'] = (1 + ret * y_fitted_rfc).cumprod()

# Evaluate VAR model
from .var_predictor import y_fitted_var

error_var = y_fitted_var - y_test_num
mse_var = sum(error_var * error_var) / len(error_var)
print(f'MSFE VAR: {mse_var}')

y_fitted_var = y_fitted_var.apply(sign_ret)
evaluate(y_fitted_var, y_test)
strats['PCA VAR'] = (1 + ret * y_fitted_var).cumprod()

# Plot
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = 'browser'
default_layout = go.Layout(font_family='Gravitas One',
                           xaxis_showgrid=False,
                           yaxis_showgrid=False,
                           showlegend=True,
                           legend_orientation='h',
                           legend_yanchor='bottom',
                           legend_y=1,
                           legend_xanchor='center',
                           legend_x=0.5)

one = 'RF Classifier'
two = 'RF Regressor'
three = 'PCA VAR'
four = 'Buy-and-Hold'

fig = go.Figure()

trace_one = go.Scatter(x=strats.index, y=strats[one], name=one)
trace_two = go.Scatter(x=strats.index, y=strats[two], name=two)
trace_three = go.Scatter(x=strats.index, y=strats[three], name=three)
trace_four = go.Scatter(x=strats.index, y=strats[four], name=four)

fig.add_trace(trace_one)
fig.add_trace(trace_two)
fig.add_trace(trace_three)
fig.add_trace(trace_four)

fig.update_layout(default_layout)
fig.show()
# save an interactive version of the graph
fig.write_html('resources/output/strats.html')
status = 'Finished!'
