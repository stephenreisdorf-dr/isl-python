# 3.6.3
# Import the libraries needed to load the BOSTON dataset 
from numpy.core.fromnumeric import var
from sklearn.datasets import load_boston
import pandas as pd

# Load the "Boston" dataset and parse into a pandas DataFrame
boston_dict = load_boston()
boston = pd.DataFrame(
    boston_dict['data']
    , columns=boston_dict['feature_names']
    , 
)
boston['MEDV'] = boston_dict['target']

# Time to fit the model
from statsmodels.api import OLS, add_constant

# Define the response and predictor
y = boston['MEDV']
# Note that statsmodels doesn't implicitly add an intercept
X = add_constant(boston[['LSTAT', 'AGE']])

# Fit the model and display the results
model = OLS(y, X)
results = model.fit()
print(results.summary())

# Define the response and predictor
y = boston['MEDV']
# Note that statsmodels doesn't implicitly add an intercept
X = add_constant(boston.drop('MEDV', axis='columns'))

# Fit the model and display the results
model = OLS(y, X)
results = model.fit()
print(results.summary())

# See the names of things that can be accessed
print(dir(results))

# Calculate VIF for each variable
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = [ variance_inflation_factor(results.model.exog, i) for i in range(results.model.exog.shape[1]) ]
print(pd.DataFrame(vif, index=X.columns, columns=['VIF']))
