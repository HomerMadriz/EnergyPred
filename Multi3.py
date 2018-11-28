# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 19:19:55 2018

@author: omar_
"""

import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from datetime import date
from datetime import datetime as dt

def agregarDia(dfFilt):
    dia=[] #serie donde coloco todo lo que voy a adicionar en la columna
    for index,row in dfFilt.iterrows():
        dia.append(date.isoweekday(dt.strptime(row['Fecha'],'%d/%m/%Y')))
    #https://pandas.pydata.org/pandas-docs/stable/merging.html
    sdia=pd.Series(dia, name='Dia') 
    dfFilt = pd.concat([dfFilt, sdia], axis=1)
    return dfFilt

def agregarZona(dfFilt):
    zona=[] #serie donde coloco todo lo que voy a adicionar en la columna
    
    dfFilt = dfFilt.sort_values(by = ['Zona de Carga'])
    dfFilt = dfFilt.reset_index(drop= True)
    n = 1
    prev = dfFilt['Zona de Carga'][0]
    for index,row in dfFilt.iterrows():
        if row['Zona de Carga'] != prev:
            n += 1
            prev = row['Zona de Carga']
        zona.append(n)
    
    #https://pandas.pydata.org/pandas-docs/stable/merging.html
    szona=pd.Series(zona, name='Zona') 
    dfFilt = pd.concat([dfFilt, szona], axis=1)
    return dfFilt
    

df2017 = pd.read_csv("Datos2017.csv", engine='python')
df2018 = pd.read_csv("Datos2018.csv", engine = 'python')


dfFilt17 = df2017[['Fecha','Hora', 'Zona de Carga', 'Precio Zonal  ($/MWh)']]
dfFilt18 = df2018[['Fecha','Hora', 'Zona de Carga', 'Precio Zonal  ($/MWh)']]

dfFilt17 = agregarDia(dfFilt17)
dfFilt18 = agregarDia(dfFilt18)

dfFilt17 = agregarZona(dfFilt17)
dfFilt18 = agregarZona(dfFilt18)

df_x_train = dfFilt17[['Hora', 'Dia', 'Zona']]
df_x_test = dfFilt18[['Hora', 'Dia', 'Zona']]

df_y_train = dfFilt17[['Precio Zonal  ($/MWh)']]
df_y_test = dfFilt18[['Precio Zonal  ($/MWh)']]

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

df_x_train = sm.add_constant(df_x_train)

df_x_test = sm.add_constant(df_x_test)

model = sm.OLS(df_y_train, df_x_train).fit()
predictions = model.predict(df_x_test)

summ = model.summary()

df_y_pred = pd.DataFrame(df_y_pred)
print(df_y_test.describe())
print(df_y_pred.describe())

df_y_test.plot()
df_y_pred.plot()
