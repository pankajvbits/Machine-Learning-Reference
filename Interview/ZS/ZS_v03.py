# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 20:12:50 2018

@author: HP 15-E015TX
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


os.chdir(r"F:\Data Science\Interview\ZS\drive-download-20180803T060231Z-001")
#Importing dataset
demog = pd.read_csv("demog.csv")
train = pd.read_csv("train.csv")
submission = pd.read_csv("submission.csv")

"""
Converting the categorical variables into continuous data
"""
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
dict_labelencode={}
dict_mean={}
dict_nanCount={}

columns = list(train.columns)
for col in columns:
   if train[col].dtype not in ['int64', 'float64']:
      index = train.columns.get_loc(col)
      print("label encoding  for -----> ", col, index)
      labelencode = LabelEncoder()
      train[col] = labelencode.fit_transform(train[col])
      dict_labelencode[col] = labelencode.classes_
      print("One hot encoding  for -----> ", col, index)                
      onehotencoder = OneHotEncoder(categorical_features = 'all')
      dummy_var = onehotencoder.fit_transform(train[col].values.reshape(-1,1)).toarray()
      valueList = dict_labelencode[col]
      i = 0 
      for value in valueList:
         print("One hot encoding  for value-----> ", value) 
         newcolumn = col + "_" + value
         print("One hot encoding  for label class -----> ", newcolumn)
         train[newcolumn] = dummy_var[:, i]
         i = i+1
      train = train.drop([col], axis=1)
   else:
      dict_mean[col] = train[col].mean()
      dict_nanCount[col] = train[col].isnull().sum()


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

for key, value in dict_nanCount.items():
   if value != 0:  
      print("Imputer ---> ", key)
      newcolumn = key+"_filled"
      train[newcolumn] = imputer.fit_transform(train[key].values.reshape(-1,1))
   else:
      pass
   
train.info()
dict_col = {}
for col in train.columns:
   dict_col[col] = train.columns.get_loc(col)

predicted = pd.DataFrame()

master_column_list = []
for col in ['Region_rural', 'Region_urban', 'Value_H', 'Value_L',
       'Value_M', 'Value_U', 'RL_filled', 'P2P_filled', 'OLV_filled',
       'RR_filled', 'DRT_filled', 'DMS_filled', 'OLA_filled', 'DEM_filled']:
   master_column_list.append(train.columns.get_loc(col))

print(master_column_list)

"""
Independent_variable_itr_list =  ['Region_rural', 'Region_urban', 'Value_H', 'Value_L',
 'Value_M', 'Value_U', 'RL_filled', 'P2P_filled', 'OLV_filled',
 'RR_filled', 'DRT_filled', 'DMS_filled', 'OLA_filled', 'DEM_filled']


for key in dict_nanCount.keys():
   if dict_nanCount[key] != 0 and key != 'P2P':
      print("At Start Null Value count --> ", train[key].isnull().sum())
      newcol = key+"_filled"
      print("For prediction of ", key, newcol)
      
      Independent_variable_itr_list.remove(newcol) 
      
      print("At Start Independent_variable_itr_list ", Independent_variable_itr_list)
      
    
      regressor = LinearRegression()   
   
      X = train.loc[:,Independent_variable_itr_list].values

      y = train.loc[:,[newcol]].values
                   
      X_train,X_test,y_train,y_test =train_test_split(X, y, test_size=0.25) 
    
      regressor.fit(X_train, y_train)
   
      predicted[key] = regressor.predict(X).tolist()
      
      train[key].fillna(predicted[key], inplace = 1)

      Independent_variable_itr_list.append(key)
      print("At end Null Value count --> ", train[key].isnull().sum())
      print("At End Independent_variable_itr_list ", Independent_variable_itr_list)
      del([regressor, X, y,X_train, y_train])
"""


Independent_variable_itr_list = []
for col in ['Region_rural', 'Region_urban', 'Value_H', 'Value_L',
       'Value_M', 'Value_U', 'RL_filled', 'P2P_filled', 'OLV_filled',
       'RR_filled', 'DRT_filled', 'DMS_filled', 'OLA_filled', 'DEM_filled']:
   Independent_variable_itr_list.append(train.columns.get_loc(col))

print(Independent_variable_itr_list)

dict_accuracy ={}   
for key in dict_nanCount.keys():
   if dict_nanCount[key] != 0 :
      print("At Start Null Value count --> ", train[key].isnull().sum())
      newcol = key+"_filled"
      print("For prediction of ", key, newcol)
      
      Independent_variable_itr_list.remove(train.columns.get_loc(newcol)) 
      
      print("At Start Independent_variable_itr_list ", Independent_variable_itr_list)
      
    
      regressor = LinearRegression()  
    
   
      X = train.iloc[:,Independent_variable_itr_list].values

      y = train.iloc[:,train.columns.get_loc(newcol)].values
                   
      X_train,X_test,y_train,y_test =train_test_split(X, y, test_size=0.25)
      
      regressor.fit(X_train, y_train)
      # Applying k-Fold Cross Validation

      accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10)
      
      dict_accuracy[key] = [accuracies.mean(), accuracies.std()]
      #print(key ,"Accuracies Mean :", accuracies.mean())
      #print(key ,"Accuracies std :", accuracies.std())
      
  
  
      predicted[key] = regressor.predict(X).tolist()
      
      train[key].fillna(predicted[key], inplace = 1)

      Independent_variable_itr_list.append(train.columns.get_loc(key))
      print("At end Null Value count --> ", train[key].isnull().sum())
      print("At End Independent_variable_itr_list ", Independent_variable_itr_list)
      del([regressor, X, y,X_train, y_train])   

   
                    