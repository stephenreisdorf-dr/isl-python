# 3.6.3
# Import the libraries needed to load the BOSTON dataset 
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

# Add interaction term
boston['LSTAT:AGE'] = boston['LSTAT'] * boston['AGE']

# Define the response and predictor
y = boston['MEDV']
# Note that statsmodels doesn't implicitly add an intercept
X = add_constant(boston[['LSTAT', 'AGE', 'LSTAT:AGE']])

# Fit the model and display the results
model = OLS(y, X)
results = model.fit()
print(results.summary())
