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

# Plot medv vs lstat
from plotly import graph_objects as go

# Create a new Figure
fig = go.Figure()
# Plot the data as is
fig.add_trace(
    go.Scatter(x=boston['LSTAT'], y=boston['MEDV'], mode='markers')
)
# Plot the predicted line
fig.add_trace(
    go.Scatter(x=boston['LSTAT'], y=results.fittedvalues)
)

# Some diagnostic plots (these are different than the book)
# READ MORE:
# https://www.statsmodels.org/stable/examples/notebooks/generated/regression_plots.html#Single-Variable-Regression-Diagnostics
from statsmodels.graphics.regressionplots import plot_fit, plot_leverage_resid2, plot_partial_residuals, plot_regress_exog
plot_regress_exog(results, "LSTAT")

# Residuals vs Fitted plot
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=results.fittedvalues
        , y=results.resid
        , mode='markers'
    )
)

# Scale - Location
import numpy as np
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=results.fittedvalues
        , y=np.sqrt(np.abs(results.resid_pearson))
        , mode='markers'
    )
)

# Residual vs Leverage
influence = results.get_influence()
leverage = influence.hat_matrix_diag
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=leverage
        , y=results.resid_pearson
        , mode='markers'
    )
)

# QQ Plot
from statsmodels.graphics.gofplots import qqplot
qqplot(results.resid_pearson)

dir(results)