# -*- coding: utf-8 -*-
"""
@author: MXX
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

#read data
#Please change file path
raw_data=pd.read_excel('C:/Users/MXX/Desktop/Marketplace Analyst/Question 3.xlsx',sheet_name='RAWDATA')

#data cleaning
#check NAN
print(raw_data.isnull().sum())

#Constant Variable
constant=[]
for i in range(1,29):
    std=raw_data.iloc[:,i].std()
    if std==0:
        constant.append(i)

raw_data = raw_data.drop(columns=['NUM_COLOR_DESC','IF_GREY','IF_UNIVRED','IF_BRAND_STORY','IF_SEASONAL_SILO_HO'])

#Categorical variables convert to dummy variables
dummy1 = pd.get_dummies(raw_data['COLOR_CD_1'],prefix='color1' )
dummy2 = pd.get_dummies(raw_data['COLOR_CD_2'],prefix='color2' )
dummy3 = pd.get_dummies(raw_data['COLOR_CD_3'],prefix='color3' )
data=raw_data.drop(columns=['SKU','COLOR_CD_1','COLOR_CD_2','COLOR_CD_3'])

data  = pd.concat([data,dummy1,dummy2,dummy3],axis = 1)

#4.	Data type transfer
dummy_names=['IF_CORE_PRICE','IF_PREMIUM_PRICE','IF_GENERAL_PRICE','IF_FW_BLACKGOLD','IF_TRIPLEWHITE',\
             'IF_BLACK_WHITE','IF_RETRO','IF_SEASONAL_SILO_SU',\
             'LAST_SEASON_INDICATOR','IF_NEW_NEW','IF_NEW_SEASONAL','IF_CARRYOVER']

data[dummy_names] = data[dummy_names].astype(np.uint8)

#correlation coefficient matrix
cor = data.corr()
cor_list = cor.iloc[:,0]

#Train test split
X_train,X_test,Y_train,Y_test = train_test_split(data.iloc[:,1:],data.iloc[:,0],train_size=.8,random_state=11)

#sm.ols
model1 = sm.OLS(Y_train, X_train).fit()
print (model1.summary())

#add constant
model2 = sm.OLS(Y_train, sm.add_constant(X_train)).fit()
print (model2.summary())

#significant variables
data_sig = data[['SALES_QTY','IF_GENERAL_PRICE','SEASON_SINCE_1ST_SKU_LAUNCH','IF_RETRO','IF_SEASONAL_SILO_SU',\
                 'NUMSKU_SAME_STYLE_IN1SSN','OMD_DAYS_SINCE_SEASON_BEGIN']]
X_train1,X_test1,Y_train1,Y_test1 = train_test_split(data_sig.iloc[:,1:],data_sig.iloc[:,0],train_size=.8,random_state=11)

model3 = sm.OLS(Y_train1, X_train1).fit()
print (model3.summary())

model4 = sm.OLS(Y_train1, sm.add_constant(X_train1)).fit()
print (model4.summary())


data_sig1 = data[['SALES_QTY','SEASON_SINCE_1ST_SKU_LAUNCH','IF_RETRO','IF_SEASONAL_SILO_SU','NUMSKU_SAME_STYLE_IN1SSN']]
X_train2,X_test2,Y_train2,Y_test2 = train_test_split(data_sig1.iloc[:,1:],data_sig1.iloc[:,0],train_size=.8,random_state=11)
model5 = sm.OLS(Y_train2, X_train2).fit()
print (model5.summary())

model6 = sm.OLS(Y_train2,sm.add_constant(X_train2)).fit()
print (model6.summary())

