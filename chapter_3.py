from numpy import load
from sklearn.datasets import load_boston
import pandas as pd

from statsmodels.api import OLS, add_constant

# Load the "Boston" dataset and parse into a pandas DataFrame
boston_dict = load_boston()
boston = pd.DataFrame(
    boston_dict['data']
    , columns=boston_dict['feature_names']
    , 
)
boston['MEDV'] = boston_dict['target']

# Describe the "Boston" data set
print(boston_dict['DESCR'])

# Define the response and predictor
y = boston['MEDV']
# Note that statsmodels doesn't 
X = add_constant(boston[['LSTAT']])

model = OLS(y, X)
results = model.fit()
results.summary()
