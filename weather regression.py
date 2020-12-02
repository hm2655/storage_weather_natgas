# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 09:48:14 2019

@author: harshit.mahajan
"""

import os
import pandas as pd 
from datetime import date 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import numpy as np 
import seaborn as sns 
from sklearn import preprocessing, svm 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

today = date.today()

path = "XXXX"
filename = "nationaldd_inventory.xlsx"

filepath = path + filename
excelFile = pd.ExcelFile(filepath)
sheetList = excelFile.sheet_names

eiaprint = excelFile.parse('bbgquery', header = None).iloc[4:,[0,1]].reset_index(drop=True)
eiaprint.columns = ['eia_date','eia_level']


degreeday = excelFile.parse('dashboard', header = None).iloc[3:,[1,2,5,6,7]].reset_index(drop=True). dropna(axis = 'rows')
degreeday.columns = ['eia_date','eia_level','hdd', 'cdd', 'holiday']
degreeday['temp'] = 65 - degreeday['hdd'] + degreeday['cdd']

counter_val = [] 
intercept = []
hdd_coef = []
cdd_coef = []
holiday_coef = []
rmse_level = []
r2_level = []

regdata = pd.DataFrame(columns = ['intercept','hdd_levels','cdd_levels', 'holiday', 'rmse_level', 'r2_level', 'counter'])

# Regression Data 
for i in range(50,500):
   df_testing = degreeday.iloc[:i, [1,2,3,4]].reset_index(drop=True)
   df_testing.head()
   testing = LinearRegression()
   testing.fit(df_testing[['hdd','cdd', 'holiday']], df_testing['eia_level'])
   predicated_levels = testing.predict(df_testing[['hdd','cdd','holiday']])
   rmse = mean_squared_error(df_testing['eia_level'], predicated_levels)
   r2 = r2_score(df_testing['eia_level'], predicated_levels)
   intercept.append(testing.intercept_)
   hdd_coef.append(testing.coef_[0])
   cdd_coef.append(testing.coef_[1])
   holiday_coef.append(testing.coef_[2]) 
   rmse_level.append(rmse) 
   r2_level.append(r2)    
   counter_val.append(i)
   print(testing.coef_)    
   print(rmse)    
   print(r2)    
   
   
regdata['intercept'] = intercept
regdata['hdd_levels'] = hdd_coef
regdata['cdd_levels'] = cdd_coef
regdata['holiday'] = holiday_coef
regdata['rmse_level'] = rmse_level
regdata['r2_level'] = r2_level
regdata['counter'] = counter_val

plt.plot(regdata['counter'], regdata['r2_level'])

plt.plot(regdata['counter'], regdata['rmse_level'])
