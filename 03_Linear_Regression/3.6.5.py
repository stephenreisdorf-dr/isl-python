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
import numpy as np

# Add interaction term
boston['LSTAT^2'] = boston['LSTAT'] ** 2

# Define the response and predictor
y = boston['MEDV']
# Note that statsmodels doesn't implicitly add an intercept
X = add_constant(boston[['LSTAT', 'LSTAT^2']])

# Fit the model and display the results
model2 = OLS(y, X)
results2 = model2.fit()
print(results2.summary())

# Build a baseline model for comparison
# Define the response and predictor
# Note that statsmodels doesn't implicitly add an intercept
X = add_constant(boston[['LSTAT']])

# Fit the model and display the results
model1 = OLS(y, X)
results1 = model1.fit()
print(results1.summary())

# Use ANOVA to compare the two models
from statsmodels.stats.anova import anova_lm
print(anova_lm(results1, results2))

# Build a degree 5 polynomial model
for i in range(3,6):
    boston['LSTAT^{}'.format(i)] = boston['LSTAT'] ** i

lstat_cols = [ col for col in boston.columns if 'LSTAT' in col ]

# Define the response and predictor
# Note that statsmodels doesn't implicitly add an intercept
X = add_constant(boston[lstat_cols])

# Fit the model and display the results
model5 = OLS(y, X)
results5 = model5.fit()
# This doesn't seem to match the books coefficients (but does match the R values)
print(results5.summary())

# Log transformation of RM
boston['log(RM)'] = np.log(boston['RM'])
# Define the response and predictor
# Note that statsmodels doesn't implicitly add an intercept
X = add_constant(boston[['log(RM)']])

# Fit the model and display the results
modellog = OLS(y, X)
resultslog = modellog.fit()
# This doesn't seem to match the books coefficients (but does match the R values)
print(resultslog.summary())
