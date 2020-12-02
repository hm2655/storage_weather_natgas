# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 10:52:52 2020

@author: harshit.mahajan
"""

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

path = "XX"
filename = "nationaldd_inventory.xlsx"

filepath = path + filename
excelFile = pd.ExcelFile(filepath)
sheetList = excelFile.sheet_names

eiaprint = excelFile.parse('bbgquery', header = None).iloc[4:,[0,1]].reset_index(drop=True)
eiaprint.columns = ['eia_date','eia_level']

degreeday = excelFile.parse('dashboard', header = None).iloc[5:,[3,4,7,8]].reset_index(drop=True).dropna(axis = 'rows')
degreeday.columns = ['eia_date','eia_level','hdd', 'cdd']
degreeday['temp'] = 65 - degreeday['hdd'] + degreeday['cdd']

counter_val = [] 
intercept = []
hdd_coef = []
cdd_coef = []
rmse_level = []
r2_level = []
predict_value = []
eia_level = []

regdata = pd.DataFrame(columns = ['intercept','hdd_levels','cdd_levels','rmse_level', 'r2_level', 'counter'])

# Regression Data 
for i in range(50,200):
   df_testing = degreeday.iloc[:i, [1,2,3,4]].reset_index(drop=True)
   df_testing.head()
   testing = LinearRegression()
   testing.fit(df_testing[['hdd','cdd']], df_testing['eia_level'])
   predicated_levels = testing.predict(df_testing[['hdd','cdd']])
   rmse = mean_squared_error(df_testing['eia_level'], predicated_levels)
   r2 = r2_score(df_testing['eia_level'], predicated_levels)
   intercept.append(testing.intercept_)
   hdd_coef.append(testing.coef_[0])
   cdd_coef.append(testing.coef_[1])
   rmse_level.append(rmse) 
   r2_level.append(r2)    
   counter_val.append(i)
   predict_value.append(df_testing.iloc[0,1] * testing.coef_[0] + df_testing.iloc[0,2] * testing.coef_[1] + testing.intercept_) 
   eia_level.append(df_testing.iloc[0,0])
   print(testing.coef_)    
   print(rmse)    
   print(r2)    
   
regdata['intercept'] = intercept
regdata['hdd_levels'] = hdd_coef
regdata['cdd_levels'] = cdd_coef
regdata['rmse_level'] = rmse_level
regdata['r2_level'] = r2_level
regdata['counter'] = counter_val
regdata['predicted_value'] = predict_value
regdata['eia_level'] = eia_level
regdata['error'] = regdata['eia_level'] - regdata['predicted_value']


plt.plot(regdata['counter'], regdata['r2_level'])
plt.plot(regdata['counter'], regdata['rmse_level'])
plt.plot(regdata['counter'], regdata['predicted_value'])
plt.plot(regdata['counter'], regdata['eia_level'])
plt.plot(regdata['counter'], regdata['error'])

plt.plot(regdata['eia_level'], regdata['predicted_value'])


###############################################################################

counter_val = [] 
intercept = []
hdd_coef = []
cdd_coef = []
rmse_level = []
r2_level = []
predict_value = []
eia_level = []
eia_date =[]

regdata = pd.DataFrame(columns = ['intercept','hdd_levels','cdd_levels', 'rmse_level', 'r2_level', 'counter'])

i=0
for i in range(0,100):
   j=i+160
   df_testing = degreeday.iloc[i:j, [0,1,2,3]].reset_index(drop=True)
   df_testing.head()
   testing = LinearRegression()
   testing.fit(df_testing[['hdd','cdd']], df_testing['eia_level'])
   predicated_levels = testing.predict(df_testing[['hdd','cdd']])
   rmse = mean_squared_error(df_testing['eia_level'], predicated_levels)
   r2 = r2_score(df_testing['eia_level'], predicated_levels)
   intercept.append(testing.intercept_)
   hdd_coef.append(testing.coef_[0])
   cdd_coef.append(testing.coef_[1])
   rmse_level.append(rmse) 
   r2_level.append(r2)    
   counter_val.append(i)
   predict_value.append(df_testing.iloc[0,2] * testing.coef_[0] + df_testing.iloc[0,3] * testing.coef_[1] + testing.intercept_) 
   eia_level.append(df_testing.iloc[0,1])
   eia_date.append(df_testing.iloc[0,0])
   print(testing.coef_)    
   print(rmse)    
   print(r2)    
   

regdata['eia_date'] = eia_date
regdata['intercept'] = intercept
regdata['hdd_levels'] = hdd_coef
regdata['cdd_levels'] = cdd_coef
regdata['rmse_level'] = rmse_level
regdata['r2_level'] = r2_level
regdata['counter'] = counter_val
regdata['predicted_value'] = predict_value
regdata['eia_level'] = eia_level
regdata['error'] = regdata['eia_level'] - regdata['predicted_value']


plt.plot(regdata['counter'], regdata['r2_level'])
plt.plot(regdata['counter'], regdata['rmse_level'])
plt.plot(regdata['eia_date'], regdata['predicted_value'], label='no_holiday')
plt.plot(regdata['eia_date'], regdata['eia_level'])
plt.legend()
