# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 15:09:35 2018

@author: omar_
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

zona = 'ACAPULCO'

df = pd.read_csv("C:/Users/omar_/Documents/ITESO/5to Semestre/Programación Avanzada/Proyecto/PreciosEnergia.csv", engine='python')
acapulco = df[df['Zona de Carga'].isin([zona])]

acapulcoco = acapulco.loc[:,['Fecha','Hora', 'Precio Zonal  ($/MWh)']]

acapulco_x = acapulcoco.loc[:, 'Hora']
acapulco_y = acapulcoco.loc[:, 'Precio Zonal  ($/MWh)']

acapulco_x_train = acapulco_x[:]
acapulco_x_test = acapulco_x[:]

acapulco_y_train = acapulco_y[:]
acapulco_y_test = acapulco_y[:]

acapulco_x_train = acapulco_x_train.as_matrix()
acapulco_x_test = acapulco_x_test.as_matrix()
acapulco_y_train = acapulco_y_train.as_matrix()

acapulco_x_train = acapulco_x_train.reshape(-1,1)
acapulco_y_train = acapulco_y_train.reshape(-1,1)
acapulco_x_test = acapulco_x_test.reshape(-1,1)

regr = linear_model.LinearRegression()

regr.fit(acapulco_x_train, acapulco_y_train)

acapulco_y_pred = regr.predict(acapulco_x_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(acapulco_y_test, acapulco_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(acapulco_y_test, acapulco_y_pred))

# Plot outputs
plt.scatter(acapulco_x_test, acapulco_y_test,  color='black')
plt.plot(acapulco_x_test, acapulco_y_pred, color='blue', linewidth=3)
plt.show()

#graficar box plot
data = acapulcoco
data.boxplot('Precio Zonal  ($/MWh)', by= 'Hora', figsize=(12,8))

plt.xticks(())
plt.yticks(())

plt.show()
