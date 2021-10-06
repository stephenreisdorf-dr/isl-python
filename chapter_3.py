# 3.6.1 (Only some libraries!)
# Libraries will be imported as needed so it's clear what they're for
# I would normally put my imports all at the top...
# ... but it makes sense to spread them help understand the code!

# 3.6.2
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
# Check out the columns!
print(boston.columns)

# Describe the "Boston" data set
print(boston_dict['DESCR'])

# Time to fit the model
from statsmodels.api import OLS, add_constant

# Define the response and predictor
y = boston['MEDV']
# Note that statsmodels doesn't implicitly add an intercept
X = add_constant(boston[['LSTAT']])

# Fit the model and display the results
model = OLS(y, X)
results = model.fit()
print(results.summary())

# Get just the coefficients
print(results.params)

# Get the 95% confidence intervals
print(results.conf_int(alpha=.95))

# Get simple predictions
X_pred = add_constant([5,10,15])
print(results.predict(X_pred))

# Get confidence and prediction intervals
print(results.get_prediction(X_pred).conf_int())
