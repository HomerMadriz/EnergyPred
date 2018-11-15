# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 16:54:12 2018

@author: ghernand
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from datetime import date
from datetime import datetime as dt


df=pd.read_csv('DatosPrecios.csv',engine='python')
#df['Dia']=[] #columna para adicionar
dia=[] #serie donde coloco todo lo que voy a adicionar en la columna
'''   
otra estrategia es duplicar la columna y luego cambiarle el valor a la categoria
dia de la semana
'''
'''for index, row in df.iterrows():
    print row["c1"], row["c2"]'''
for index,row in df.iterrows():
    dia.append(date.isoweekday(dt.strptime(row['Fecha'],'%d/%m/%Y')))
#https://pandas.pydata.org/pandas-docs/stable/merging.html
sdia=pd.Series(dia, name='Dia') 
df = pd.concat([df, sdia], axis=1)



