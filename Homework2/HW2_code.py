# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 00:28:02 2018

@author: Kuo
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor


def washData(df):
    for i in range(len(df)):
        tmp = df.iloc[i][3:]
        idx = [ n for n, word in enumerate(tmp) if type(word) == str]
        if len(idx) == 0 or idx == 0: continue
        
        for j in idx:
            tmp = df.iloc[i][3:]
            idx = [ n for n, word in enumerate(tmp) if type(word) == str]
            
            noHead = False
            noEnd = False
            
            headIdx = j-1
            endIdx = j+1

            #第一個就有缺失
            if j == 0: noHead = True        
            while endIdx in idx: endIdx += 1                
            if endIdx > 23: noEnd = True
                
            if noHead:
                df.iat[i, j+3] = tmp[endIdx]
                continue
            if noEnd:
                df.iat[i, j+3] = tmp[headIdx]
                continue            
            df.iat[i, j+3] = (tmp[headIdx]+tmp[endIdx])/2
        
    return df

def formatData(data):
    #1. 先把資料拉成 18*61*24 = 18 *1464
    for i in range(0, len(data), 18):    
        d = data[i:i+18]
        if i == 0:
            d = d.drop([r'日期',r'測站'], axis=1) 
            result = d
            continue
        else:
            d = d.drop([r'日期',r'測站', r'測項'], axis=1)
        result = pd.concat([result.reset_index(drop=True), d.reset_index(drop=True)], axis=1)
    result = result.set_index(r'測項')
    
    #2. 把 18*1464 的資料，弄成 time series data
    totalLen = len(result.iloc[0])
    x = list()
    x_18 = list()
    y = result.iloc[9][6:].tolist()
    
    for i in range(totalLen-6):
        x_18.append(result.iloc[: ,i : (i+6)].values) #as_matrix()
    
    result = result.iloc[9].tolist()
    for i in range(totalLen-6):
        x.append(result[i : (i+6)])
        
    x = np.array(x, dtype=float)
    x_18 = np.array(x_18, dtype=float)
        
    return x, x_18, y
    
    
df = pd.read_excel(r'106年新竹站_20180309.xls')
df = df[4914:] #只取10~12月份:training(10、11), testing(12)

df = df.replace('NR', 0) #'NR' value replace with 0
df.fillna(value=0, inplace=True) #np.nan 補 0
df = washData(df) #補齊缺失值、無效值

trainData = df.iloc[:1098]
testData = df.iloc[1098:]

#data prepare
train_x, train_x_18, train_y = formatData(trainData)
test_x, test_x_18, test_y = formatData(testData) 

train_x_18 = train_x_18.reshape(-1, 18*6)
test_x_18 = test_x_18.reshape(-1, 18*6)

# =============================================================================
#  Linear Regression
# =============================================================================
# consider only PM2.5
regPM = LinearRegression().fit(train_x, train_y)
pred_y_PM = regPM.predict(test_x) #testing data
print('[Linear Regression] 只有考慮PM2.5的MAE = ', mean_absolute_error(test_y, pred_y_PM))

#consider 18 attributes
regAll = LinearRegression().fit(train_x_18, train_y)
pred_y_All = regAll.predict(test_x_18)
print('[Linear Regression] 考慮18個屬性的MAE = ', mean_absolute_error(test_y, pred_y_All))


# =============================================================================
#  Random Forest Regression
# =============================================================================
# consider only PM2.5
regrPM = RandomForestRegressor(max_depth=8, random_state=10, n_estimators=200)
regrPM.fit(train_x, train_y)
pred_y_PM = regrPM.predict(test_x)
print('[Random Forest Regression] 只有考慮PM2.5的MAE = ', mean_absolute_error(test_y, pred_y_PM))

#consider 18 attributes
regrAll = RandomForestRegressor(max_depth=8, random_state=10, n_estimators=200)
regrAll.fit(train_x_18, train_y)
pred_y_All = regrAll.predict(test_x_18)
print('[Random Forest Regression] 考慮18個屬性的MAE = ', mean_absolute_error(test_y, pred_y_All))




