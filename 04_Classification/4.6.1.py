import pandas as pd
import requests
from io import StringIO
from plotly import graph_objects as go

# Read Smarket data from some guys repo
smarket_url = 'https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/Smarket.csv'
smarket_response = requests.get(smarket_url)
smarket_df = pd.read_csv(
    StringIO(smarket_response.text)
    , index_col=0
)

print(smarket_df.shape)
print(smarket_df.info())
print(smarket_df.describe())

print(smarket_df.corr())

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=smarket_df.index
        , y=smarket_df['Volume']
        , mode='markers'
    )
)
fig.show()
