import pandas as pd

data = pd.read_csv('resources/input/final_dataset.csv', index_col = 'Date')

# Independent Variables
X = data.iloc[:,1:]

# Dependent Variable
y = data['CRIX'].diff()/ data['CRIX'].shift(1)
y.rename('ret', inplace = True)
y.dropna(inplace = True)

# Comparable Model: PCA & Vector Autoregression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Create the PCA Index
standard_X = StandardScaler().fit_transform(X.values) # Standardize before PCA
pca = PCA(n_components = 1)
PCA_X = pca.fit_transform(standard_X)
PCA_X = pd.DataFrame(data = PCA_X, columns = ['PCA'])
PCA_X.index = X.index

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

# Stationary Tests
def stationary(var, desc): 
    print(f"'{desc}' is stationary: {adfuller(var)[0] < adfuller(var)[4]['1%']}.")

stationary(PCA_X, 'PCA X') # PCA X is not stationary
X['delta PCA'] = PCA_X.diff()
X.dropna(inplace = True)
stationary(X['delta PCA'], 'delta PCA X') # delta PCA X is stationary
stationary(y, 'Ret') # Ret is stationary

Xy = pd.concat([y, X['delta PCA']], axis = 1)
Xy_train = Xy.loc[:'2019-03-04'] # Train data 
Xy_test = Xy.loc['2019-02-28':'2020-07-25'] # Test data to predict y from 2019-03-05 to 2020-07-25

# ignore some non-important warnings
import warnings
warnings.filterwarnings("ignore", message="A date index has been ")
warnings.filterwarnings("ignore", message="The default dtype ")

model = VAR(Xy_train)
result = model.fit(maxlags = 15, ic = 'bic')
result.summary()
          
# extract estimated coefficients
const = result.coefs_exog[0][0]
coefs = pd.DataFrame()
for i in range(4, -1, -1):
    coefs = coefs.append(pd.Series(result.coefs[i][0]), ignore_index = True)
coefs.rename({0 : 'ret', 1 : 'delta PCA'}, axis = 1, inplace = True)

# compute the predicted return
y_fitted_var = pd.Series(index = Xy.loc['2019-03-05':'2020-07-25'].index)
for i in range(len(Xy_test)-5):
    y_fitted_var.iloc[i] = (Xy_test.iloc[i:(i+5)].values * coefs).sum().sum() + const
