# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 20:12:47 2019

@author: ASUSNB
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from statsmodels.tsa.arima_model import ARMA,ARMAResults,ARIMA,ARIMAResults
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf # for determining (p,q) orders
% matplotlib inline
df = pd.read_csv('NYPD_Motor_Vehicle_Collisions.csv')
df['DATE'] = df['DATE'].astype('datetime64[ns]')

brooklyn= df[df['BOROUGH'] == 'BROOKLYN']

brooklyn.describe()


df5= brooklyn['UNIQUE KEY'].groupby(df.DATE).agg('count')
df5= pd.DataFrame(df5)


df5['UNIQUE KEY'].plot(figsize(12,8))
ax = df5[['UNIQUE KEY']].plot(figsize=(12,5),legend = False)
ax.autoscale(axis='both' , tight = True)
ax.set(xlabel = 'Date' , ylabel = '$Total Accidents')
plt.show()
from statsmodels.tsa.stattools import adfuller


def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")
        
adf_test(df5['UNIQUE KEY'])


## Data is  stationary 

from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df5)
pylplot.show()

from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf

plot_pacf(df5, lags=50)
pyplot.show() 
#(1, 1, 1   )

plot_acf(df5, lags=50)
pyplot.show()

train = df5.iloc[:1771]
test = df5.iloc[1771:]

    
model = ARIMA(train['UNIQUE KEY'],order=(7,1,5))
results = model.fit()
results.summary()

start = len(train)
end = len(train) + len(test) -1

predictions = results.predict(start = start , end = end, typ = 'levels').rename('ARIMA(7,1,5) Predictions')

test['UNIQUE KEY'].plot(legend = True , figsize = (12,8))
predictions.plot(legend = True)

fcast = results.predict( end = len(df5)+209,typ='levels').rename('ARIMA(7,1,5) Forecast')
fcast = pd.DataFrame(fcast)
fcast.reset_index(inplace=True)

fcast = fcast[2530:]

fcast.info()
fcast = fcast.set_index('index')
del fcast['level_0']

fcast.head(30)

export_csv = fcast.to_csv ('2019.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path


