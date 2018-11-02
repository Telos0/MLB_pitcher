# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 22:35:02 2018

@author: telos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data2015 = pd.read_csv('2015.txt')
data2016 = pd.read_csv('2016.txt')
data2017 = pd.read_csv('2017.txt')

raw_data = pd.concat((data2015, data2016, data2017)).reset_index()

data = pd.concat((data2015, data2016, data2017)).reset_index()
del data['index']
del data['Name']
del data['Rk']
del data['W-L%']
del data['G']
del data['GS']
del data['GF']
del data['CG']
del data['SHO']
del data['SV']
del data['H']
del data['R']
del data['ER']
del data['HR']
del data['BB']
del data['IBB']
del data['SO']
del data['HBP']
del data['BK']
del data['WP']
del data['BF']
del data['ERA+']
data.drop(2712, inplace = True)
data = data[data['IP']>=50]
data.drop(883, inplace = True)
data.drop(1790, inplace = True)

team = {x:t for t,x in enumerate(list(data['Tm'].unique()))}
lg = {x:t for t,x in enumerate(list(data['Lg'].unique()))}

data['Tm'].replace(team, inplace = True)
data['Lg'].replace(lg, inplace = True)

data_copy = data.copy()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

y = data['ERA']
del data['ERA']
X = data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

SC = StandardScaler()
SC.fit(X_train)
X_train_std = SC.transform(X_train)
X_test_std = SC.transform(X_test)

#X_train_std = X_train
#X_test_std = X_test

from sklearn.linear_model import SGDRegressor

SGD = SGDRegressor(max_iter = 10000, random_state = 0)
SGD.fit(X_train_std, y_train)
a = SGD.predict(X_test_std)
SGD2 = SGDRegressor(loss = 'epsilon_insensitive' ,max_iter = 10000, random_state = 0)
SGD2.fit(X_train_std, y_train)
b = SGD2.predict(X_test_std)

def rmsle(y, y_):
    log1 = np.nan_to_num(np.log(y + 1))
    log2 = np.nan_to_num(np.log(y_ + 1))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))

def rmse(y, y_):
    return np.sqrt(np.mean((y-y_)**2))

#print(rmsle(y_test, a))
print('SGDRegressor error : %f' %float(np.mean(abs(y_test-a))))
print('SGDRegressor2 error : %f' %float(np.mean(abs(y_test-b))))

from sklearn.linear_model import LinearRegression, Ridge, Lasso

LR = LinearRegression()
LR.fit(X_train_std, y_train)
pr = LR.predict(X_test_std)

print('LinearRegression error : %f' %float(np.mean(abs(y_test-pr))))

RI = Ridge()
RI.fit(X_train_std, y_train)
ri = RI.predict(X_test_std)

print('Ridge error : %f' %float(np.mean(abs(y_test-ri))))

LA = Lasso()
LA.fit(X_train_std, y_train)
la = LA.predict(X_test_std)

print('Lasso error : %f' %float(np.mean(abs(y_test-la))))

import matplotlib.patches as mpatches

plt.plot(range(len(y_test)),y_test)
plt.plot(range(len(y_test)),pr, c = 'red')
plt.xlabel('index')
plt.ylabel('ERA')
red_patch = mpatches.Patch(color='red', label='LR predict')
blue_patch = mpatches.Patch(color='blue', label='Real')
plt.legend([red_patch, blue_patch],['LR predict', 'Real'])
plt.show()


plt.plot(range(len(y_test)),y_test)
plt.plot(range(len(y_test)),ri, c = 'green')
plt.xlabel('index')
plt.ylabel('ERA')
red_patch = mpatches.Patch(color='green', label='RI predict')
blue_patch = mpatches.Patch(color='blue', label='Real')
plt.legend([red_patch, blue_patch],['RI predict', 'Real'])
plt.show()

plt.plot(range(len(y_test)),y_test)
plt.plot(range(len(y_test)),la, c = 'black')
plt.xlabel('index')
plt.ylabel('ERA')
red_patch = mpatches.Patch(color='black', label='LA predict')
blue_patch = mpatches.Patch(color='blue', label='Real')
plt.legend([red_patch, blue_patch],['LA predict', 'Real'])
plt.show()


# train set acc
LR = LinearRegression()
LR.fit(X_train_std, y_train)
pr = LR.predict(X_train_std)

print('LinearRegression error : %f' %float(np.mean(abs(y_train-pr))))

RI = Ridge()
RI.fit(X_train_std, y_train)
ri = RI.predict(X_train_std)

print('Ridge error : %f' %float(np.mean(abs(y_train-ri))))

LA = Lasso()
LA.fit(X_train_std, y_train)
la = LA.predict(X_train_std)

print('Lasso error : %f' %float(np.mean(abs(y_train-la))))


plt.plot(range(len(y_train)),y_train)
plt.plot(range(len(y_train)),pr, c = 'red')
plt.xlabel('train_index')
plt.ylabel('ERA')
red_patch = mpatches.Patch(color='red', label='LR predict')
blue_patch = mpatches.Patch(color='blue', label='Real')
plt.legend([red_patch, blue_patch],['LR predict', 'Real'])
plt.show()


plt.plot(range(len(y_train)),y_train)
plt.plot(range(len(y_train)),ri, c = 'green')
plt.xlabel('train_index')
plt.ylabel('ERA')
red_patch = mpatches.Patch(color='green', label='RI predict')
blue_patch = mpatches.Patch(color='blue', label='Real')
plt.legend([red_patch, blue_patch],['RI predict', 'Real'])
plt.show()

plt.plot(range(len(y_train)),y_train)
plt.plot(range(len(y_train)),la, c = 'black')
plt.xlabel('train_index')
plt.ylabel('ERA')
red_patch = mpatches.Patch(color='black', label='LA predict')
blue_patch = mpatches.Patch(color='blue', label='Real')
plt.legend([red_patch, blue_patch],['LA predict', 'Real'])
plt.show()