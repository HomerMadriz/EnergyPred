# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 12:51:18 2018

@author: omar_
"""

import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from datetime import date
from datetime import datetime as dt
import matplotlib.pyplot as plt

df2017 = pd.read_csv("Datos2017.csv", engine='python')
df2018 = pd.read_csv("Datos2018.csv", engine = 'python')

frames = [df2017, df2018]
df1718 = pd.concat(frames)

dfFilt = df1718[['Fecha','Hora', 'Precio Zonal  ($/MWh)']]

dfFilt = dfFilt.reset_index(drop= True)
dia=[] #serie donde coloco todo lo que voy a adicionar en la columna

for index,row in dfFilt.iterrows():
    dia.append(date.isoweekday(dt.strptime(row['Fecha'],'%d/%m/%Y')))
#https://pandas.pydata.org/pandas-docs/stable/merging.html
sdia=pd.Series(dia, name='Dia') 
dfFilt = pd.concat([dfFilt, sdia], axis=1)

df_x = dfFilt[['Hora', 'Dia']]
df_y = dfFilt[['Precio Zonal  ($/MWh)']]

df_x_train = df_x[:884762]
df_x_test = df_x[884762:]

df_y_train = df_y[:884762]
df_y_test = df_y[884762:]

regr = linear_model.LinearRegression()
regr.fit(df_x_train, df_y_train)
df_y_pred = regr.predict(df_x_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(df_y_test, df_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(df_y_test, df_y_pred))

df_y_pred = pd.DataFrame(df_y_pred)
print(df_y_test.describe())
print(df_y_pred.describe())

plt.scatter(range(0,30), df_y_test, color='blue')
plt.scatter(range(0,30), df_y_pred, color='green')
plt.show()

df_x_train = sm.add_constant(df_x_train)

model = sm.OLS(df_y_train, df_x_train).fit()
predictions = model.predict(df_x_test)

summ = model.summary()
print(summ)
