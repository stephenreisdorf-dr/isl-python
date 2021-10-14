import pandas as pd
import requests
from io import StringIO
import statsmodels.api as sm

# Read Smarket data from some guys repo
smarket_url = 'https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/Smarket.csv'
smarket_response = requests.get(smarket_url)
smarket_df = pd.read_csv(
    StringIO(smarket_response.text)
    , index_col=0
)

# Fit Logistic Regression model
y=smarket_df['Direction'].map(lambda x: 1 if x == 'Up' else 0)
X=sm.add_constant(
    smarket_df.drop([ 'Today', 'Direction', 'Year' ], axis='columns')
)
model = sm.Logit(y, X)
results = model.fit()

# Display model results
print(results.summary())

# Display coefficients
print(results.params)

# Display first 10 predictions
results.predict(X)[:10]

y_pred = results.predict(X) >= .5
y_pred_map = y_pred.map(lambda x: 'Up' if x else 'Down')

# Display confusion matrix
print(results.pred_table())
