import pandas as pd
import requests
from io import StringIO
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

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
print(results.predict(X)[:10])

y_pred = results.predict(X) >= .5
y_pred_map = y_pred.map(lambda x: 'Up' if x else 'Down')

# Display confusion matrix
print(confusion_matrix(y, y_pred))

# Calculate training error rate
print(accuracy_score(y, y_pred))

# Retrain the model on a subset of the data
X_train = sm.add_constant(
    smarket_df[ smarket_df['Year'] < 2005 ]
    .drop([ 'Today', 'Direction', 'Year' ], axis='columns')
)
y_train = (
    smarket_df
    [ smarket_df['Year'] < 2005 ]['Direction']
    .map(lambda x: 1 if x == 'Up' else 0)
)

X_test = sm.add_constant(
    smarket_df[ smarket_df['Year'] >= 2005 ]
    .drop([ 'Today', 'Direction', 'Year' ], axis='columns')
)
y_test = (
    smarket_df
    [ smarket_df['Year'] >= 2005 ]['Direction']
    .map(lambda x: 1 if x == 'Up' else 0)
)
model = sm.Logit(y_train, X_train)
results = model.fit()

y_pred = results.predict(X_test) >= 0.5
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# Retrain the model on a subset of the data with just Lag1 and Lag2
X_train = sm.add_constant(
    smarket_df[ smarket_df['Year'] < 2005 ]
    [['Lag1', 'Lag2']]
)
y_train = (
    smarket_df
    [ smarket_df['Year'] < 2005 ]['Direction']
    .map(lambda x: 1 if x == 'Up' else 0)
)

X_test = sm.add_constant(
    smarket_df[ smarket_df['Year'] >= 2005 ]
    [['Lag1', 'Lag2']]
)
y_test = (
    smarket_df
    [ smarket_df['Year'] >= 2005 ]['Direction']
    .map(lambda x: 1 if x == 'Up' else 0)
)
model = sm.Logit(y_train, X_train)
results = model.fit()

y_pred = results.predict(X_test) >= 0.5
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred))

results.predict(sm.add_constant([[1.2,1.5],[1.1,-0.8]]))
