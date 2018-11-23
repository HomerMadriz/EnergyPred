# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 12:08:01 2018

@author: omar_
"""

import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df2017 = pd.read_csv("Datos2017.csv", engine='python')
df2018 = pd.read_csv("Datos2018.csv", engine = 'python')

dfFilt17 = df2017[['Fecha','Hora', 'Precio Zonal  ($/MWh)']]
dfFilt18 = df2018[['Fecha','Hora', 'Precio Zonal  ($/MWh)']]

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