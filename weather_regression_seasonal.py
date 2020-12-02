# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 11:23:51 2020

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

degreeday = excelFile.parse('dashboard', header = None).iloc[5:,[2,3,4,7,8,9]].reset_index(drop=True).dropna(axis = 'rows')
degreeday.columns = ['weeknum','eia_date','eia_level','hdd', 'cdd','holiday']
degreeday['temp'] = 65 - degreeday['hdd'] + degreeday['cdd']

counter_val = [] 
intercept = []
hdd_coef = []
cdd_coef = []
rmse_level = []
r2_level = []
predict_value = []
eia_level = []
holiday_coef =[]
eia_date = []

regdata_holiday = pd.DataFrame(columns = ['eia_date','intercept','hdd_levels','cdd_levels','holiday','rmse_level', 'r2_level', 'counter'])

i=0
for i in range(0,200):
   df_testing = degreeday.iloc[i:, [0,1,2,3,4,5]].reset_index(drop=True)
   weeknum = df_testing.iloc[0,0] 
   base_data = df_testing.iloc[0,]  
   print(weeknum)
   
   #week_list = [weeknum+49,weeknum+50,weeknum+51,weeknum+52,weeknum+53,
   #             weeknum+101,weeknum+102,weeknum+103,weeknum+104,weeknum+105,
    #            weeknum+153,weeknum+154,weeknum+155,weeknum+156,weeknum+157,
     #           weeknum+205,weeknum+206,weeknum+207,weeknum+208,weeknum+209,
      #          weeknum+257,weeknum+258,weeknum+259,weeknum+260,weeknum+261]
    
   #row_list = [49,50,51,52,53,101,102,103,104,105,153,154,155,156,157,205,206,207,208,209,257,258,259,260,261]
   row_list = [49,50,51,52,53,101,102,103,104,105,153,154,155,156,157,205,206,207,208,209,257,258,259,260,261,54,106,158,210,262]
   df_testing = df_testing.iloc[row_list, [0,1,2,3,4,5]].reset_index(drop=True)
   df_testing.head()
   testing = LinearRegression()
   testing.fit(df_testing[['hdd','cdd','holiday']], df_testing['eia_level'])
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
   predict_value.append(base_data.iloc[3] * testing.coef_[0] + base_data.iloc[4] * testing.coef_[1] + base_data.iloc[5] * testing.coef_[2] + testing.intercept_) 
   eia_level.append(base_data.iloc[2])
   eia_date.append(base_data.iloc[1])
   print(testing.coef_)    
   print(rmse)    
   print(r2)    
   

regdata_holiday['eia_date'] = eia_date
regdata_holiday['intercept'] = intercept
regdata_holiday['hdd_levels'] = hdd_coef
regdata_holiday['cdd_levels'] = cdd_coef
regdata_holiday['holiday'] = holiday_coef
regdata_holiday['rmse_level'] = rmse_level
regdata_holiday['r2_level'] = r2_level
regdata_holiday['counter'] = counter_val
regdata_holiday['predicted_value'] = predict_value
regdata_holiday['eia_level'] = eia_level
regdata_holiday['error'] = np.sqrt(((regdata_holiday['eia_level'] - regdata_holiday['predicted_value'])**2)/100)


plt.plot(regdata_holiday['counter'], regdata_holiday['r2_level'])
plt.plot(regdata_holiday['counter'], regdata_holiday['rmse_level'])
plt.plot(regdata_holiday['eia_date'], regdata_holiday['predicted_value'])
plt.plot(regdata_holiday['eia_date'], regdata_holiday['eia_level'])
plt.plot(regdata_holiday['eia_date'], regdata_holiday['error'])
sum(regdata_holiday['error'])/100


############################################################################

degreeday = excelFile.parse('dashboard', header = None).iloc[5:,[2,3,4,7,8]].reset_index(drop=True).dropna(axis = 'rows')
degreeday.columns = ['weeknum','eia_date','eia_level','hdd', 'cdd']
degreeday['temp'] = 65 - degreeday['hdd'] + degreeday['cdd']

counter_val = [] 
intercept = []
hdd_coef = []
cdd_coef = []
rmse_level = []
r2_level = []
predict_value = []
eia_level = []
holiday_coef =[]
eia_date = []
hdd_data=[]

regdata = pd.DataFrame(columns = ['eia_date','intercept','hdd_levels','cdd_levels','rmse_level', 'r2_level', 'counter'])

i=0
for i in range(0,200):
   df_testing = degreeday.iloc[i:, [0,1,2,3,4]].reset_index(drop=True)
   weeknum = df_testing.iloc[0,0] 
   base_data = df_testing.iloc[0,]  
   print(weeknum)
   
   #week_list = [weeknum+49,weeknum+50,weeknum+51,weeknum+52,weeknum+53,
   #             weeknum+101,weeknum+102,weeknum+103,weeknum+104,weeknum+105,
    #            weeknum+153,weeknum+154,weeknum+155,weeknum+156,weeknum+157,
     #           weeknum+205,weeknum+206,weeknum+207,weeknum+208,weeknum+209,
      #          weeknum+257,weeknum+258,weeknum+259,weeknum+260,weeknum+261]
    
   row_list = [49,50,51,52,53,101,102,103,104,105,153,154,155,156,157,205,206,207,208,209,257,258,259,260,261,54,106,158,210,262]
   df_testing = df_testing.iloc[row_list, [0,1,2,3,4]].reset_index(drop=True)
   df_testing.head()
   testing = LinearRegression()
   testing.fit(df_testing[['hdd','cdd']], df_testing['eia_level'])
   predicated_levels = testing.predict(df_testing[['hdd','cdd']])
   rmse = mean_squared_error(df_testing['eia_level'], predicated_levels)
   r2 = r2_score(df_testing['eia_level'], predicated_levels)
   intercept.append(testing.intercept_)
   hdd_coef.append(testing.coef_[0])
   cdd_coef.append(testing.coef_[1])
   hdd_data.append(base_data['hdd'] + base_data['cdd'])
   rmse_level.append(rmse) 
   r2_level.append(r2)    
   counter_val.append(i)
   predict_value.append(base_data.iloc[3] * testing.coef_[0] + base_data.iloc[4] * testing.coef_[1] + testing.intercept_) 
   eia_level.append(base_data.iloc[2])
   eia_date.append(base_data.iloc[1])
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
regdata['error'] = np.sqrt(((regdata['eia_level'] - regdata['predicted_value'])**2)/100)
regdata['hdd_data']=hdd_data


plt.plot(regdata['counter'], regdata['r2_level'])
plt.plot(regdata['counter'], regdata['rmse_level'])
plt.plot(regdata['eia_date'], regdata['predicted_value'])
plt.plot(regdata['eia_date'], regdata['eia_level'])
plt.plot(regdata['eia_date'], regdata['error'])
plt.scatter(regdata['hdd_data'], regdata['error'])

sum(regdata['error'])/200
sum(regdata_holiday['error'])/200


import statsmodels.tsa.stattools as ts 
ts.adfuller(regdata['error'],1)

plt.plot(regdata['eia_level'], regdata['predicted_value'])


plt.plot(regdata['eia_date'], regdata['predicted_value'], color= 'red', label = 'no_holiday_season')
plt.plot(regdata_holiday['eia_date'], regdata_holiday['predicted_value'], color='blue', label = 'holiday_seasonal')
plt.plot(regdata['eia_date'], regdata['eia_level'], color = 'black', label = 'eia_data')
plt.legend()
plt.plot(regdata['eia_date'], regdata['eia_level'])
