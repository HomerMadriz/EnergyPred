# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 18:17:49 2018

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

zona = 'ZAMORA'
dfZona1718 = df1718[df1718['Zona de Carga'].isin([zona])]

dfFilt = dfZona1718[['Fecha','Hora', 'Precio Zonal  ($/MWh)']]

dfFilt = dfFilt.reset_index(drop= True)
dia=[] #serie donde coloco todo lo que voy a adicionar en la columna

for index,row in dfFilt.iterrows():
    dia.append(date.isoweekday(dt.strptime(row['Fecha'],'%d/%m/%Y')))
#https://pandas.pydata.org/pandas-docs/stable/merging.html
sdia=pd.Series(dia, name='Dia') 
dfFilt = pd.concat([dfFilt, sdia], axis=1)

df_x = dfFilt[['Hora', 'Dia']]
df_y = dfFilt[['Precio Zonal  ($/MWh)']]

df_x_train = df_x[:8760]
df_x_test = df_x[8760:]

df_y_train = df_y[:8760]
df_y_test = df_y[8760:]

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

df_x = sm.add_constant(df_x)

model = sm.OLS(df_y, df_x).fit()
predictions = model.predict(df_x)

summ = model.summary()

# Plot outputs
#plt.scatter(df_x_test[['Hora']], df_y_test,  color='black')
plt.plot(df_y_test, df_y_pred, color='blue', linewidth=1)
plt.plot(df_x_test[['Dia']], df_y_test, color='green', linewidth=1)
plt.show()


##graficar box plot
##data = dfFilt
##data.boxplot('Precio Zonal  ($/MWh)', by= 'Hora', figsize=(12,8))
##
##plt.show()
#
## Plot outputs
#plt.scatter(df_x_test[['Dia']], df_y_test,  color='black')
#plt.plot(df_x_test[['Dia']], df_y_pred, color='red', linewidth=1)
#plt.show()
#
###graficar box plot
##data = dfFiltrado
##data.boxplot('Precio Zonal  ($/MWh)', by= 'Dia', figsize=(12,8))
##
##plt.show()

print(summ)
