# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 14:12:56 2018

@author: telos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data2015_1 = pd.read_csv('2015.txt')
data2015_1.drop(883, axis = 0, inplace = True)

def chg(x):
    if type(x) != str:
        return x
    else:
        return float(x.replace('%', ''))*0.01
    
data2015_2 = pd.read_csv('2015ratio.txt')
del data2015_2['Rk']
del data2015_2['IP']
del data2015_2['Tm']
del data2015_2['Age']
del data2015_2['PAu']
del data2015_2['Name']
data2015_2 = data2015_2.applymap(chg)

data2015_3 = pd.read_csv('2015pitch.txt')
del data2015_3['Rk']
del data2015_3['IP']
del data2015_3['Tm']
del data2015_3['Age']
del data2015_3['PAu']
del data2015_3['Name']
data2015_3 = data2015_3.applymap(chg)

data2015 = pd.concat((data2015_1, data2015_2, data2015_3), axis = 1)


data2016_1 = pd.read_csv('2016.txt')
data2016_1.drop(906, axis = 0, inplace = True)

data2016_2 = pd.read_csv('2016ratio.txt')
del data2016_2['Rk']
del data2016_2['IP']
del data2016_2['Tm']
del data2016_2['Age']
del data2016_2['PAu']
del data2016_2['Name']
data2016_2 = data2016_2.applymap(chg)

data2016_3 = pd.read_csv('2016pitch.txt')
del data2016_3['Rk']
del data2016_3['IP']
del data2016_3['Tm']
del data2016_3['Age']
del data2016_3['PAu']
del data2016_3['Name']
data2016_3 = data2016_3.applymap(chg)

data2016 = pd.concat((data2016_1, data2016_2, data2016_3), axis = 1)

data2017_1 = pd.read_csv('2017.txt')
data2017_1.drop(921, axis = 0, inplace = True)

data2017_2 = pd.read_csv('2017ratio.txt')
del data2017_2['Rk']
del data2017_2['IP']
del data2017_2['Tm']
del data2017_2['Age']
del data2017_2['PAu']
del data2017_2['Name']
data2017_2 = data2017_2.applymap(chg)

data2017_3 = pd.read_csv('2017pitch.txt')
del data2017_3['Rk']
del data2017_3['IP']
del data2017_3['Tm']
del data2017_3['Age']
del data2017_3['PAu']
del data2017_3['Name']
data2017_3 = data2017_3.applymap(chg)

data2017 = pd.concat((data2017_1, data2017_2, data2017_3), axis = 1)

#2018 test
data2018_1 = pd.read_csv('2018.txt')
data2018_1.drop(755, axis = 0, inplace = True)

data2018_2 = pd.read_csv('2018ratio.txt')
del data2018_2['Rk']
del data2018_2['IP']
del data2018_2['Tm']
del data2018_2['Age']
del data2018_2['PAu']
del data2018_2['Name']
data2018_2 = data2018_2.applymap(chg)

data2018_3 = pd.read_csv('2018pitch.txt')
del data2018_3['Rk']
del data2018_3['IP']
del data2018_3['Tm']
del data2018_3['Age']
del data2018_3['PAu']
del data2018_3['Name']
data2018_3 = data2018_3.applymap(chg)

data2018 = pd.concat((data2018_1, data2018_2, data2018_3), axis = 1)


data = pd.concat((data2015, data2016, data2017), axis = 0).reset_index()

delete = ('Name', 'index', 'Rk', 'W-L%', 'G',
          'GS', 'GF', 'CG', 'SHO', 'SV', 'H', 'R', 'ER', 'HR', 'BB', 'IBB',
       'SO', 'HBP', 'BK', 'WP', 'BF', 'ERA+', 'SO/W', 'SO-BB%', 'X/H%', 'Opp', 'DP',
       '%', 'PA', 'Pit', 'Str', 'Str%','I/Str', 'AS/Str', 'I/Bll', 'AS/Pit', '30c',
       '30s', '02c', '02s', '02h', 'L/SO', 'S/SO', '3pK', '4pW', 'Pitu', 'Stru')
for i in delete:
    del data[i]

team = {x:t for t,x in enumerate(list(data['Tm'].unique()))}
lg = {x:t for t,x in enumerate(list(data['Lg'].unique()))}
data['Tm'].replace(team, inplace = True)
data['Lg'].replace(lg, inplace = True)
data = data[data['IP']>=50]

data_copy = data.copy()

#2018 data setting
data2018['Tm'].replace(team, inplace = True)
data2018['Lg'].replace(lg, inplace = True)
data2018 = data2018[data2018['IP']>=50]
y_2018 = data2018['ERA']
del data2018['ERA']
X_2018 = data2018

#plotting and corr
'''fig, axes = plt.subplots(nrows=8,ncols=4)
fig.set_size_inches(12, 28)
lc = list(data.columns)
lc.remove('ERA')
axes[0][0].scatter(data['Age'], data['ERA'])
axes[0][0].set_xlabel('Age')
axes[0][0].set_ylabel('ERA')
z = np.polyfit(data['Age'], data['ERA'], 1)
p = np.poly1d(z)
axes[0][0].plot(data['Age'],p(data['Age']), 'r')
for num, x in enumerate(lc):
    if num == 0:
        continue
    i, j = divmod(num, 4)
    axes[i][j].scatter(data[x], data['ERA'])
    axes[i][j].set_xlabel(x)
    axes[i][j].set_ylabel('ERA')
    z = np.polyfit(data[x], data['ERA'], 1)
    p = np.poly1d(z)
    axes[i][j].plot(data[x],p(data[x]), 'r')
plt.show()
corr = data.corr()'''

#error measure functions
def rmsle(y, y_):
    log1 = np.nan_to_num(np.log(y + 1))
    log2 = np.nan_to_num(np.log(y_ + 1))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))

def rmse(y, y_):
    return np.sqrt(np.mean((y-y_)**2))

#start learning
y = data['ERA']
del data['ERA']
X = data

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

SGD = SGDRegressor(max_iter = 10000, random_state = 0)
LR = LinearRegression()
RI = Ridge(alpha = 1)
LA = Lasso()
SVM = SVR(degree = 2, kernel = 'rbf', C = 1e3, gamma = 0.01)
DT = DecisionTreeRegressor(random_state = 0)

catagory_list = list(X.columns)
variable_combination = []

import itertools

for i in range(29,len(catagory_list)+1):
    for subset in itertools.combinations(catagory_list, i):
        variable_combination.append(list(subset))

print('combination 완성 : %d' %len(variable_combination))

SGD_acc = []
LR_acc = []
RI_acc = []
LA_acc = []
SVM_acc = []
DT_acc = []

#RI_result = []

for j, combination in enumerate(variable_combination):
    
    #combination
    X_com_train = X_train[combination]
    X_com_test = X_test[combination]
    X_com_2018 = X_2018[combination]
    
    #standardscale
    SC = StandardScaler()
    SC.fit(X_com_train)
    X_com_train_std = SC.transform(X_com_train)
    X_com_test_std = SC.transform(X_com_test)
    #X_com_2018_std = SC.transform(X_com_2018)
    
    '''#SGD
    SGD.fit(X_com_train_std, y_train)
    sd = SGD.predict(X_com_test_std)
    SGD_acc.append(float(np.mean(abs(y_test-sd))))
    
    #LinearRegression
    LR.fit(X_com_train_std, y_train)
    pr = LR.predict(X_com_test_std)
    LR_acc.append(float(np.mean(abs(y_test-pr))))'''
    
    #Ridge
    RI.fit(X_com_train_std, y_train)
    ri = RI.predict(X_com_test_std)
    RI_acc.append(float(np.mean(abs(y_test-ri))))
    #ri2 = RI.predict(X_com_2018_std)
    #RI_result.append(ri2)
    
    '''#Lasso
    LA.fit(X_com_train_std, y_train)
    la = LA.predict(X_com_test_std)
    LA_acc.append(float(np.mean(abs(y_test-la))))
    
    #SVM
    SVM.fit(X_com_train_std, y_train)
    svm = SVM.predict(X_com_test_std)
    SVM_acc.append(float(np.mean(abs(y_test-svm))))
    
    #DT
    DT.fit(X_com_train_std, y_train)
    dt = DT.predict(X_com_test_std)
    DT_acc.append(float(np.mean(abs(y_test-dt))))'''
    
    msg ='\r%d 개 완료'%(j+1)
    print(' '*len(msg),end='')
    print(msg,end='')


#return result
print('')
#print('SGD min error : %f' %np.min(SGD_acc)) 
print('RI min error : %f' %np.min(RI_acc))
#print('LA min error : %f' %np.min(LA_acc))
#print('LR min error : %f' %np.min(LR_acc))
#print('SVM min error : %f' %np.min(SVM_acc))
#print('DT min error : %f' %np.min(DT_acc))

#plotting and corr
import matplotlib.patches as mpatches

'''#2018 data
result = RI_result[np.argmin(RI_acc)]
plt.plot(range(len(y_2018)), y_2018, c = 'blue')
plt.plot(range(len(y_2018)), result, c = 'green')
plt.title('2018_index')
plt.ylabel('2018_ERA')
red_patch = mpatches.Patch(color='green', label='RI predict')
blue_patch = mpatches.Patch(color='blue', label='Real')
plt.legend([red_patch, blue_patch],['RI predict', 'Real'])
plt.show()
print('RI_2018_error : %f' %np.mean(abs(y_2018 - result)))'''

'''#SGD plot
num = variable_combination[np.argmin(SGD_acc)]
mat1, mat2 = X_train[num], X_test[num]
SC = StandardScaler()
SC.fit(mat1)
mat1 = SC.transform(mat1)
mat2 = SC.transform(mat2)

SGD.fit(mat1, y_train)
sdd = SGD.predict(mat1)
plt.figure(figsize = (10,5))
plt.subplot(121)
plt.plot(range(len(y_train)), y_train, c = 'blue')
plt.plot(range(len(y_train)), sdd, c = 'green')
plt.title('train_index')
plt.ylabel('ERA')
red_patch = mpatches.Patch(color='green', label='SGD predict')
blue_patch = mpatches.Patch(color='blue', label='Real')
plt.legend([red_patch, blue_patch],['SGD predict', 'Real'])
plt.subplot(122)
sda = SGD.predict(mat2)
plt.plot(range(len(y_test)), y_test, c = 'blue')
plt.plot(range(len(y_test)), sda, c = 'green')
plt.title('test_index')
plt.ylabel('ERA')
red_patch = mpatches.Patch(color='green', label='SGD predict')
blue_patch = mpatches.Patch(color='blue', label='Real')
plt.legend([red_patch, blue_patch],['SGD predict', 'Real'])
plt.show()
print('SGD_train_error : %f' %np.mean(abs(y_train - sdd)))
print('SGD_test_error : %f' %np.mean(abs(y_test - sda)))'''

#RI plot
num = variable_combination[np.argmin(RI_acc)]
mat1, mat2 = X_train[num], X_test[num]
SC = StandardScaler()
SC.fit(mat1)
mat1 = SC.transform(mat1)
mat2 = SC.transform(mat2)

RI.fit(mat1, y_train)
ri1 = RI.predict(mat1)
plt.figure(figsize = (10,5))
plt.subplot(121)
plt.plot(range(len(y_train)), y_train, c = 'blue')
plt.plot(range(len(y_train)), ri1, c = 'black')
plt.title('train_index')
plt.ylabel('ERA')
red_patch = mpatches.Patch(color='black', label='RI predict')
blue_patch = mpatches.Patch(color='blue', label='Real')
plt.legend([red_patch, blue_patch],['RI predict', 'Real'])
plt.subplot(122)
ri2 = RI.predict(mat2)
plt.plot(range(len(y_test)), y_test, c = 'blue')
plt.plot(range(len(y_test)), ri2, c = 'black')
plt.title('test_index')
plt.ylabel('ERA')
red_patch = mpatches.Patch(color='black', label='RI predict')
blue_patch = mpatches.Patch(color='blue', label='Real')
plt.legend([red_patch, blue_patch],['RI predict', 'Real'])
plt.show()
print('RI_train_error : %f' %np.mean(abs(y_train - ri1)))
print('RI_test_error : %f' %np.mean(abs(y_test - ri2)))

'''#LR plot
num = variable_combination[np.argmin(LR_acc)]
mat1, mat2 = X_train[num], X_test[num]
SC = StandardScaler()
SC.fit(mat1)
mat1 = SC.transform(mat1)
mat2 = SC.transform(mat2)

LR.fit(mat1, y_train)
lr1 = LR.predict(mat1)
plt.figure(figsize = (10,5))
plt.subplot(121)
plt.plot(range(len(y_train)), y_train, c = 'blue')
plt.plot(range(len(y_train)), lr1, c = 'yellow')
plt.title('train_index')
plt.ylabel('ERA')
red_patch = mpatches.Patch(color='yellow', label='LR predict')
blue_patch = mpatches.Patch(color='blue', label='Real')
plt.legend([red_patch, blue_patch],['LR predict', 'Real'])
plt.subplot(122)
lr2 = LR.predict(mat2)
plt.plot(range(len(y_test)), y_test, c = 'blue')
plt.plot(range(len(y_test)), lr2, c = 'yellow')
plt.title('test_index')
plt.ylabel('ERA')
red_patch = mpatches.Patch(color='yellow', label='LR predict')
blue_patch = mpatches.Patch(color='blue', label='Real')
plt.legend([red_patch, blue_patch],['LR predict', 'Real'])
plt.show()
print('LR_train_error : %f' %np.mean(abs(y_train - lr1)))
print('LR_test_error : %f' %np.mean(abs(y_test - lr2)))'''

'''#SVM plot
num = variable_combination[np.argmin(SVM_acc)]
mat1, mat2 = X_train[num], X_test[num]
SC = StandardScaler()
SC.fit(mat1)
mat1 = SC.transform(mat1)
mat2 = SC.transform(mat2)

SVM.fit(mat1, y_train)
sv1 = SVM.predict(mat1)
plt.figure(figsize = (10,5))
plt.subplot(121)
plt.plot(range(len(y_train)), y_train, c = 'blue')
plt.plot(range(len(y_train)), sv1, c = 'red')
plt.title('train_index')
plt.ylabel('ERA')
red_patch = mpatches.Patch(color='red', label='SVM predict')
blue_patch = mpatches.Patch(color='blue', label='Real')
plt.legend([red_patch, blue_patch],['SVM predict', 'Real'])
plt.subplot(122)
sv2 = SVM.predict(mat2)
plt.plot(range(len(y_test)), y_test, c = 'blue')
plt.plot(range(len(y_test)), sv2, c = 'red')
plt.title('test_index')
plt.ylabel('ERA')
red_patch = mpatches.Patch(color='red', label='SVM predict')
blue_patch = mpatches.Patch(color='blue', label='Real')
plt.legend([red_patch, blue_patch],['SVM predict', 'Real'])
plt.show()
print('SVM_train_error : %f' %np.mean(abs(y_train - sv1)))
print('SVM_test_error : %f' %np.mean(abs(y_test - sv2)))'''
