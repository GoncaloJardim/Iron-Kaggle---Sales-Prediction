# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 14:44:21 2021

@author: crocs
"""

import pandas as pd
import numpy as np
import scipy.stats as st
#  import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import xgboost
from sklearn.preprocessing import MinMaxScaler
import pickle

data = pd.read_csv(r'C:\Users\crocs\OneDrive - Universidade de Lisboa\Ambiente de Trabalho\Gonçalo\College&Courses\Data Analytics- IronHack\Course Data Analytics\Classes\Classes_Week7\Class_03Dec\IronKaggle\Sales.csv')

data = pd.concat([data, pd.get_dummies(data['Day_of_week'])], axis=1)

data["Date"] = pd.to_datetime(data["Date"])

data = pd.concat([data, pd.get_dummies(data["State_holiday"])], axis=1)


data = data[data['Open'] == 1]

data["date_year"] = data["Date"].dt.year
data["date_month"] = data["Date"].dt.month
data["date_week"] = data["Date"].dt.weekofyear
data["date_day"] = data["Date"].dt.day
data = pd.concat([data, pd.get_dummies(data["date_year"], prefix="Year")], axis=1)
data = pd.concat([data, pd.get_dummies(data["date_month"], prefix="Month")], axis=1)
data = pd.concat([data, pd.get_dummies(data["date_week"], prefix="Week")], axis=1)
data = pd.concat([data, pd.get_dummies(data["date_day"], prefix="Day")], axis=1)


#data = data.drop([data[(data["c"] ==1) & (data["Open"] == 1)]])

#data.drop(["c"], inplace = True)
X = data.drop(columns=["True_index","Day_of_week", "Date", "State_holiday","Sales"], axis=1)

y = data["Sales"]


#First Train:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

xgb_reg = xgboost.XGBRegressor()
    
xgb_reg.fit(X_train, y_train)


#Second train with min max scaler:

normalizer = MinMaxScaler()


X_train_normalized= normalizer.fit(X_train ).transform(X_train)
X_test_normalized = normalizer.fit(X_train).transform(X_test)

X_train_normalized = pd.DataFrame(X_train_normalized , columns = X_train.columns)
X_test_normalized = pd.DataFrame(X_test_normalized , columns = X_test.columns)


xgb_reg = xgboost.XGBRegressor()

xgb_reg.fit(X_train_normalized, y_train)



pred = xgb_reg.predict(X_test_normalized)


print("The score is : ",xgb_reg.score(X_test_normalized , y_test))

print("The overfitting is : ",(xgb_reg.score(X_train_normalized , y_train) -xgb_reg.score(X_test_normalized , y_test)))

#%%

pickle.dump(xgb_reg, open("model.p", "wb")) 