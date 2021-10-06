'''
  Python Lab Review: Chapter 3
  7/13/2021
  Based on the code from https://nbviewer.jupyter.org/github/JWarmenhoven/ISL-python/blob/master/Notebooks/Chapter%203.ipynb#3.1-Simple-Linear-Regression
  Other references provided in code.
  
'''

#3.6.1 Libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import axes3d 
import seaborn as sns

from sklearn.preprocessing import scale
import sklearn.linear_model as skl_lm
from sklearn.metrics import mean_squared_error, r2_score
import sklearn.datasets as sk_d
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as stats
from patsy import dmatrices

#%matplotlib inline
plt.style.use('seaborn-white')


# Load Dataset
Boston_X, Boston_y = sklearn.datasets.load_boston(return_X_y=True)
Boston_Names = getattr(sklearn.datasets.load_boston(),"feature_names")

Boston_X = pd.DataFrame(Boston_X)
Boston_X.columns = Boston_Names

print(Boston_X.head())

Boston_y = pd.DataFrame(Boston_y)


#3.6.2 Simple Linear Regression:
# MedV ~ LSTAT


# SKLearn....................... Not a fan
# Fit Model
X = np.array(Boston_X["LSTAT"]).reshape(-1,1)
slr_model = skl_lm.LinearRegression()
slr_model.fit(X,Boston_y)

# Review Results
# 34.6
print(slr_model.intercept_)
# -0.95
print(slr_model.coef_)
#R^2: 0.544
print(slr_model.score(X,Boston_y))


# # Get Confidence Interval for the Prediction
# # (https://medium.com/swlh/ds001-linear-regression-and-confidence-interval-a-hands-on-tutorial-760658632d99)
slr_pred = slr_model.predict(Boston_X)

def pred_interval(prediction,y_test,test_predictions,p=.95):
# '''
# Get a prediction interval for a linear regression.
# INPUTS:
# - Single prediction,
# - y_test
# - All test set predictions,
# - Prediction interval threshold (default = .95)
# OUTPUT:
# - Prediction interval for single prediction
# '''
  #Get Standard Deviation of y_test
  sum_errs = np.sum((y_test - test_predictions)**2)
  stdev = np.sqrt(1 / (len(y_test)-2) * sum_errs)

  #Get Interval from SD
  one_minus_p = 1 - p
  ppf_lookup = 1 - (one_minus_p / 2)
  z_score = stats.norm.ppf(ppf_lookup)
  interval = z_score * stdev

  lower, upper = prediction - interval, prediction + interval

  return lower, prediction, upper

slr_lower = []
slr_upper = []

for i in slr_pred:
  lower, pred, upper = pred_interval(i,Boston_y,slr_pred)

  slr_lower.append(lower)
  slr_upper.append(upper)


slr_lower = np.array(slr_lower)[:,0]
slr_upper = np.array(slr_upper)[:,0]


plt.fill_between(np.arange(0,len(Boston_y),1),np.array(slr_upper),np.array(slr_lower),color = 'b',label = 'Confidence Interval')
plt.plot(slr_pred,label='Linear Regression')
plt.plot(Boston_y,'k',label='Actual Values')
plt.xlabel('% Households with Low Socioeconomic Status')
plt.ylabel('Median House Value')
plt.title('Household Income by LStat with a 95% CI')
plt.legend()
plt.show()
plt.clf()





# Statsmodels.................... MUCH Better
Boston_Data = pd.concat((Boston_X,Boston_y),axis =1)
Boston_Names = np.append(Boston_Names,"MDEV")
Boston_Data.columns = Boston_Names

stlr_model = smf.ols(formula='MDEV ~ LSTAT',data = Boston_Data)
results = stlr_model.fit()

print(results.summary())

#Use 'dir' to find what pieces of information are stored
dir(results)

#use getattr() to access specific pieces of information
getattr(results,'params')

#Get Confidence interval for coefficients
results.conf_int(alpha = .05)

#Make prediction
Xnew = pd.DataFrame([5,10,15])
Xnew.columns = ["LSTAT"]
results.predict(Xnew)

# Diagnostic Plots
# (https://towardsdatascience.com/going-from-r-to-python-linear-regression-diagnostic-plots-144d1c4aa5a)
fig, axs = plt.subplots(2,2)

#   Residual vs Fitted
residuals = results.resid
fitted = results.fittedvalues
smoothed_rf = lowess(residuals,fitted)
top3_rf = abs(residuals).sort_values(ascending = False)[:3]


axs[0,0].scatter(fitted,residuals,edgecolors = 'k',facecolors = 'none')
axs[0,0].plot(smoothed_rf[:,0],smoothed_rf[:,1],color = 'r')
axs[0,0].set_ylabel('Residuals')
axs[0,0].set_xlabel('Fitted Values')
axs[0,0].set_title('Residuals vs Fitted')
axs[0,0].plot([min(fitted),max(fitted)],[0,0],color = 'k',linestyle = ':',alpha = .3)

for i in top3_rf.index:
  axs[0,0].annotate(i,xy=(fitted[i],residuals[i]))
  

#   Normal QQ
sorted_student_residuals = pd.Series(results.get_influence().resid_studentized_internal)
sorted_student_residuals.index = results.resid.index
sorted_student_residuals = sorted_student_residuals.sort_values(ascending = True)
df = pd.DataFrame(sorted_student_residuals)
df.columns = ['sorted_student_residuals']
df['theoretical_quantiles'] = stats.probplot(df['sorted_student_residuals'],dist='norm',fit=False)[0]
rankings = abs(df['sorted_student_residuals']).sort_values(ascending = False)
top3_qq = rankings[:3]

x = df['theoretical_quantiles']
y = df['sorted_student_residuals']

axs[1,0].scatter(x,y,edgecolor='k',facecolor='none')
axs[1,0].set_title('Normal Q-Q')
axs[1,0].set_ylabel('Standardized Residuals')
axs[1,0].set_xlabel('Theoretical Quantiles')
axs[1,0].plot([np.min([x,y]),np.max([x,y])],[np.min([x,y]),np.max([x,y])], color = 'r', ls = '--')

for i in top3_qq.index:
  axs[1,0].annotate(i,xy=(df['theoretical_quantiles'].loc[i],df['sorted_student_residuals'].loc[i]))
  

#   Scale-Location
student_residuals = results.get_influence().resid_studentized_internal
sqrt_student_residuals = pd.Series(np.sqrt(np.abs(student_residuals)))
sqrt_student_residuals.index = results.resid.index
smoothed_sl = lowess(sqrt_student_residuals,fitted)
top3_sl = abs(sqrt_student_residuals).sort_values(ascending = False)[:3]

axs[0,1].scatter(fitted,sqrt_student_residuals,edgecolors = 'k',facecolors='none')
axs[0,1].plot(smoothed_sl[:,0],smoothed_sl[:,1],color = 'r')
axs[0,1].set_ylabel('$\sqrt{|Studentized \ Residuals|}$')
axs[0,1].set_xlabel('Fitted Values')
axs[0,1].set_title('Scale-Location')
axs[0,1].set_ylim(0,max(sqrt_student_residuals)+0.1)
for i in top3_sl.index:
  axs[0,1].annotate(i,xy=(fitted[i],sqrt_student_residuals[i]))
  
  
#   Residual vs Leverage
def Leverage(fitted_model, student_residuals = None,\
            leverage = None, ax = None):
  """
  Parameters
  ---------------------------------------------------------
  fitted_model: A fitted linear regression model from the statsmodels package.
                Class: <statsmodels.regression.linear_model.OLS>
  student_residuals: A pandas series of the internally studentized residuals.
  ax: A specific matplotlib axis. Used if creating subplots
  
  Returns
  ---------------------------------------------------------
  ax: A matplotlib axis object
  The approach for coding the Cook's D lines comes from:
  https://emredjan.github.io/blog/2017/07/11/emulating-r-plots-in-python/
  
  By: Jason Sadowski
  Date: 2019-11-19
  """
  if isinstance(student_residuals,type(None)):
    student_residuals = pd.Series(fitted_model\
                           .get_influence().resid_studentized_internal)
    student_residuals.index = fitted_model.resid.index
  if isinstance(leverage,type(None)): 
    leverage = fitted_model.get_influence().hat_matrix_diag
  df = pd.DataFrame(student_residuals)
  df.columns = ['student_residuals']
  df['leverage'] = leverage
  sorted_student_resid = abs(df['student_residuals'])\
                          .sort_values(ascending = False)
  top3 = sorted_student_resid[:3]
  smoothed = lowess(student_residuals,leverage)
  if isinstance(ax,type(None)):
    fig, ax = plt.subplots()
  x = df['leverage']
  y = df['student_residuals']
  xpos = max(x)+max(x)*0.05  
  ax.scatter(x, y, edgecolors = 'k', facecolors = 'none')
  ax.plot(smoothed[:,0],smoothed[:,1],color = 'r')
  ax.set_ylabel('Studentized Residuals')
  ax.set_xlabel('Leverage')
  ax.set_title('Residuals vs. Leverage')
  ax.set_ylim(min(y)-min(y)*0.15,max(y)+max(y)*0.15)
  ax.set_xlim(-0.01,max(x)+max(x)*0.05)

  cooksx = np.linspace(min(x), xpos, 50)
  p = len(fitted_model.params)
  poscooks1y = np.sqrt((p*(1-cooksx))/cooksx)
  poscooks05y = np.sqrt(0.5*(p*(1-cooksx))/cooksx)
  negcooks1y = -np.sqrt((p*(1-cooksx))/cooksx)
  negcooks05y = -np.sqrt(0.5*(p*(1-cooksx))/cooksx)
  
  ax.plot(cooksx,poscooks1y,label = "Cook's Distance", ls = ':', color = 'r')
  ax.plot(cooksx,poscooks05y, ls = ':', color = 'r')
  ax.plot(cooksx,negcooks1y, ls = ':', color = 'r')
  ax.plot(cooksx,negcooks05y, ls = ':', color = 'r')
  ax.plot([0,0],ax.get_ylim(), ls=":", color = 'k', alpha = 0.3)
  ax.plot(ax.get_xlim(), [0,0], ls=":", color = 'k', alpha = 0.3)
  ax.annotate('1.0', xy = (xpos, poscooks1y[-1]), color = 'r')
  ax.annotate('0.5', xy = (xpos, poscooks05y[-1]), color = 'r')
  ax.annotate('1.0', xy = (xpos, negcooks1y[-1]), color = 'r')
  ax.annotate('0.5', xy = (xpos, negcooks05y[-1]), color = 'r')
  ax.legend()
  for val in top3.index:
    ax.annotate(val,xy=(x.loc[val],y.loc[val]))
  return(ax)

Leverage(results, student_residuals = None,\
            leverage = None, ax = axs[1,1])


plt.show()
plt.clf()


#3.6.3 Multiple Linear Regression:
# MDEV ~ LSTAT + AGE
stlr_model = smf.ols(formula='MDEV ~ LSTAT + AGE',data = Boston_Data)
results = stlr_model.fit()

print(results.summary())

# MDEV ~ .
#Rather than typing all variables out...
all_columns = "+".join((Boston_Names[0:13]))
my_formula = "MDEV~" + all_columns
stlr_model = smf.ols(formula=my_formula,data = Boston_Data)
results = stlr_model.fit()

print(results.summary())

#Use 'dir' to find what pieces of information are stored
dir(results)

#use getattr() to access specific pieces of information
getattr(results,'params')

#Get Confidence interval for coefficients
results.conf_int(alpha = .05)

#Make prediction
Xnew = pd.DataFrame([5,10,15])
Xnew.columns = ["LSTAT"]
results.predict(Xnew)

#Calculate variance inflation factors (https://etav.github.io/python/vif_factor_python.html)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(Boston_X.values,i) for i in range(Boston_X.shape[1])]
vif["features"] = Boston_X.columns

vif.round(1)


#3.6.4 Interaction Terms
stlr_model = smf.ols(formula='MDEV ~ LSTAT * AGE',data = Boston_Data)
results = stlr_model.fit()

print(results.summary())


#3.6.5 Non-linear Transformations of the Predictors
stlr_model = smf.ols(formula='MDEV ~ LSTAT + np.power(LSTAT,2)',data = Boston_Data)
results = stlr_model.fit()

print(results.summary())

#quantify extent to which quadratic model is superior
stlr_model2 = smf.ols(formula='MDEV ~ LSTAT',data = Boston_Data)
results2 = stlr_model2.fit()
sm.stats.anova_lm(results2,results)
