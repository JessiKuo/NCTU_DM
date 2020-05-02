# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 15:35:13 2018

@author: Kuo
Data Mining(Homework 3):Cross Validation
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

def K_fold_CV(k, data, target):
    data = data.values
    target = target.values
    target = target.reshape(-1, 1)
    
    subsetSize = (int)(len(data)/10)
    idx = np.zeros(len(data))
    cnt = 1
    for i in range(subsetSize, len(data), subsetSize):
        idx[i:i+subsetSize] = cnt
        cnt += 1
    idx[-1] = 9
    
    acc = []
    
    for i in range(k):
        trainIdx = np.where(idx!=i)[0]
        testIdx = np.where(idx==i)[0]
        
        train_x = data[trainIdx,:]
        test_x = data[testIdx,:]
        
        train_y = target[trainIdx]
        test_y = target[testIdx]
        
        gb = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.05, \
                                        max_features=30, max_depth = 10, random_state = 0)
        gb.fit(train_x, train_y)
        acc.append(gb.score(test_x, test_y))
    return acc
  
if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    df.loc[df['income'] == ' <=50K', 'income'] = 0
    df.loc[df['income'] == ' >50K', 'income'] = 1
    
    df.loc[df['sex'] == ' Male', 'sex'] = 0
    df.loc[df['sex'] == ' Female', 'sex'] = 1
    
#    df.drop(['fnlwgt', 'education', 'marital_status', 'native_country',\
#             'relationship', 'workclass', 'capital_gain',\
#             'capital_loss', 'race'],axis=1, inplace=True)
    
    data = df.iloc[:,0:14]
    target = df.iloc[:,[14]]
    
    data = pd.get_dummies(data, columns=["occupation", 'workclass', 'education',\
                                         'marital_status', 'relationship', 'race',\
                                         'native_country'])
    
#    scaler = MinMaxScaler()
#    data = scaler.fit_transform(data)
    
    acc = K_fold_CV(10, data, target)
    
    for i in range(len(acc)):
        print('k = ', (i+1), ' acc = ', acc[i])
    print('===')
    print('平均 acc = ', sum(acc)/len(acc))
    
    
    