# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 19:20:04 2020

@author: NEXTEL
"""

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#reading the data from your files
data = pd.read_csv("advertising.csv")
data.head()

#visualization of the data
fig, axs = plt.subplots(1,3, sharey = True)
data.plot(kind="scatter",x = 'TV', y = 'Sales', ax = axs[0],figsize=(14,7))
data.plot(kind="scatter",x = 'Radio', y = 'Sales', ax = axs[1])
data.plot(kind="scatter",x = 'Newspaper', y = 'Sales', ax = axs[2])

#creating x and y for linear regression(transforming dataset)
feature_cols =['TV']
X = data[feature_cols]
y = data.Sales


#importing linear regression algo for simple linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, y)
 
#get intercept and co-ef
print(lr.intercept_)
print(lr.coef_)

#y=a+bx

result = 6.974821488229891+0.05546477*50
print(result)


#create dataframe with min and max value of the table

X_new = pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
X_new.head()

#predicting using the min and max vales
preds = lr.predict(X_new) 
preds 

#how to plot least squared line to get best fit line
data.plot(kind="scatter",x='TV',y = 'Sales')
plt.plot(X_new,preds,c = 'red',linewidth=3)

#summary and confidence interval of the model
#stats model (ordinary least square)
import statsmodels.formula.api as smf
lm = smf.ols(formula = 'Sales ~ TV', data = data).fit()
lm.conf_int()

#finding the probability values
lm.pvalues

#finding the r-squared vales
lm.rsquared


#multi-linear regresssion
feature_cols =['TV','Radio','Newspaper']
X = data[feature_cols]
y = data.Sales

lr =LinearRegression()
lr.fit(X, y)

print(lr.intercept_)
print(lr.coef_)

#summary and confidence interval of the model
lm = smf.ols(formula = 'Sales ~ TV+Radio+Newspaper', data = data).fit()
lm.conf_int()
lm.summary()














