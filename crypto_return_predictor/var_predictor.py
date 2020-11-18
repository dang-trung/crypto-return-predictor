#!/usr/bin/env python3
"""Predicts Crypto Returns using VAR model.

Since the features are sentiment indicators, they are often
highly-correlated. According to Brown & Cliff (2004), it's conceptually
appealing to extract a common component of the indicators as their first
Principal Component. This common component is then fitted into a VAR
timeseries model to check if it could be good predictor of returns."""
import pandas as pd

data = pd.read_csv('resources/input/final_dataset.csv', index_col='Date')

# Independent Variables
X = data.iloc[:, 1:]

# Dependent Variable
y = data['CRIX'].diff() / data['CRIX'].shift(1)
y.rename('ret', inplace=True)
y.dropna(inplace=True)

# PCA & Vector Autoregression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Create the Sentiment Index (First PC of all features)
standard_X = StandardScaler().fit_transform(X.values)  # Standardize before PCA
pca = PCA(n_components=1)
PCA_X = pca.fit_transform(standard_X)
PCA_X = pd.DataFrame(data=PCA_X, columns=['PCA'])
PCA_X.index = X.index

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller


def stationary(var, desc):
    """
    Print if it is true that a data series is stationary.
    Parameters
    ----------
    var : array-like, 1-d
        The data series to test stationarity.
    desc : str
        Name of the variable.

    Returns
    -------
    NoneType
        None

    """
    print(
        f"'{desc}' is stationary: "
        f"{adfuller(var)[0] < adfuller(var)[4]['1%']}.")


stationary(PCA_X, 'PCA X')  # PCA X is not stationary
X['delta PCA'] = PCA_X.diff()
X.dropna(inplace=True)
stationary(X['delta PCA'], 'delta PCA X')  # delta PCA X is stationary
stationary(y, 'Ret')  # Ret is stationary

# Fit stationary variables to VAR time series model
Xy = pd.concat([y, X['delta PCA']], axis=1)
# Train data
Xy_train = Xy.loc[:'2019-03-04']
# Test data to predict y from 2019-03-05 to 2020-07-25
Xy_test = Xy.loc[
          '2019-02-28':'2020-07-25']

# Ignore non-important warnings
import warnings

warnings.filterwarnings("ignore", message="A date index has been ")
warnings.filterwarnings("ignore", message="The default dtype ")

model = VAR(Xy_train)
result = model.fit(maxlags=15, ic='bic')
result.summary()

# Extract estimated coefficients
const = result.coefs_exog[0][0]
coefs = pd.DataFrame()
for i in range(4, -1, -1):
    coefs = coefs.append(pd.Series(result.coefs[i][0]), ignore_index=True)
coefs.rename({0: 'ret', 1: 'delta PCA'}, axis=1, inplace=True)

# Predict returns by fit the provided features in the test sample
y_fitted_var = pd.Series(index=Xy.loc['2019-03-05':'2020-07-25'].index)
for i in range(len(Xy_test) - 5):
    y_fitted_var.iloc[i] = (Xy_test.iloc[
                            i:(i + 5)].values * coefs).sum().sum() + const
