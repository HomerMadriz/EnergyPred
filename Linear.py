# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 17:34:11 2018

@author: omar_
"""

import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df2017 = pd.read_csv("Datos2017.csv", engine='python')
df2018 = pd.read_csv("Datos2018.csv", engine = 'python')
zona = 'ZAMORA'
dfZona17 = df2017[df2017['Zona de Carga'].isin([zona])]
dfZona18 = df2018[df2018['Zona de Carga'].isin([zona])]

dfFilt17 = dfZona17[['Fecha','Hora', 'Precio Zonal  ($/MWh)']]
dfFilt18 = dfZona18[['Fecha','Hora', 'Precio Zonal  ($/MWh)']]

df_x_train = dfFilt17[['Hora']]
df_y_train = dfFilt17[['Precio Zonal  ($/MWh)']]

df_x_test = dfFilt18[['Hora']]
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

# Plot outputs
plt.scatter(df_x_test, df_y_test,  color='black')
plt.plot(df_x_test, df_y_pred, color='blue', linewidth=3)
plt.show()

#graficar box plot
data = dfFilt17
data.boxplot('Precio Zonal  ($/MWh)', by= 'Hora', figsize=(12,8))

plt.show()


"""
dfFiltrado = dfFiltrado.reset_index(drop= True)
dia=[] #serie donde coloco todo lo que voy a adicionar en la columna

for index,row in dfFiltrado.iterrows():
    dia.append(date.isoweekday(dt.strptime(row['Fecha'],'%d/%m/%Y')))
#https://pandas.pydata.org/pandas-docs/stable/merging.html
sdia=pd.Series(dia, name='Dia') 
dfFiltrado = pd.concat([dfFiltrado, sdia], axis=1)

print(dfFiltrado.head())"""
