#https://www.analyticsvidhya.com/blog/2018/09/multivariate-time-series-guide-forecasting-modeling-python-codes/
#https://datascienceschool.net/view-notebook/ceed009866404f7bbfc7e494336c218b/

import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

#read the data
df = pd.read_csv("BS_test.csv")
features_li = ['JS', 'MM', 'Permits']
features = df[features_li]
features.index = df['Date Time']
data = features.dropna()

#check the dtypes
data.dtypes

#missing value treatment
cols = data.columns
for j in cols:
    for i in range(0,len(data)):
       if data[j][i] == -200:
           data[j][i] = data[j][i-1]

#checking stationarity
#since the test works for only 12 variables, I have randomly dropped
#in the next iteration, I would drop another and check the eigenvalues
# johan_test_temp = data.drop([ 'CO(GT)'], axis=1)
coint_johansen(data,-1,1).eig


#creating the train and validation set
train = data[:int(0.8*(len(data)))]
valid = data[int(0.8*(len(data))):]

#fit the model
model = VAR(endog=train)
model_fit = model.fit()

# make prediction on validation
prediction = model_fit.forecast(model_fit.y, steps=len(valid))


#converting predictions to dataframe
pred = pd.DataFrame(index=range(0,len(prediction)),columns=[cols])
for j in range(0,13):
    for i in range(0, len(prediction)):
       pred.iloc[i][j] = prediction[i][j]

#check rmse
for i in cols:
    print('rmse value for', i, 'is : ', sqrt(mean_squared_error(pred[[i]], valid[i])))


#make final predictions; VAR (multivariate)
model = VAR(endog=data)
model_fit = model.fit()
yhat = model_fit.forecast(model_fit.y, steps=1)
print(yhat)

model_fit.plot_forecast(24)

#make ARIMAX predictions (univariate)
model_ARIMA = ARIMA(endog=data['MM'], order=[1,1,0])
results = model_ARIMA.fit()
print(results.summary())
# yhat = results.forecast(model_fit.y, steps=1)
# print(yhat)
