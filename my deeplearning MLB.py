# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 16:03:46 2018

@author: telos
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections

data = pd.read_csv('baseball_reference_2016_clean.csv')
data = data.fillna(data['attendance'].mean())
del data['Unnamed: 0']
del data['date']
del data['start_time']
del data['game_hours_dec']
del data['home_team_win']
del data['home_team_loss']
del data['home_team_outcome']

del data['away_team_errors']
del data['away_team_hits']
del data['home_team_errors']
del data['home_team_hits']

data['run-run'] = abs(data['away_team_runs']-data['home_team_runs'])
#data['run+run'] = abs(data['away_team_runs']-data['home_team_runs'])
del data['away_team_runs']
del data['home_team_runs']
del data['total_runs']

def did(x):
    if 1<=x<=2:
        return 2
    elif 2<x<=5:
        return 1
    else:
        return 0

data['run-run'] = data['run-run'].map(did)


team = {x:t for t,x in enumerate(list(set(data['away_team'].unique())))}
field_type = {x:t for t,x in enumerate(list(set(data['field_type'].unique())))}
venue = {x:t for t,x in enumerate(list(set(data['venue'].unique())))}
sky = {x:t for t,x in enumerate(list(set(data['sky'].unique())))}
wind_direction = {x:t for t,x in enumerate(list(set(data['wind_direction'].unique())))}
season = {x:t for t,x in enumerate(list(set(data['season'].unique())))}
game_type = {x:t for t,x in enumerate(list(set(data['game_type'].unique())))}
day_of_week = {'Sunday':0, 'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6}

data['away_team'].replace(team, inplace = True)
data['home_team'].replace(team, inplace = True)
data['field_type'].replace(field_type, inplace = True)
data['venue'].replace(venue, inplace = True)
data['sky'].replace(sky, inplace = True)
data['wind_direction'].replace(wind_direction, inplace = True)
data['season'].replace(season, inplace = True)
data['game_type'].replace(game_type, inplace = True)
data['day_of_week'].replace(day_of_week, inplace = True)

#copy
data_copy = data.copy()

#start
y = pd.DataFrame(data['run-run'])
y = y.merge(pd.get_dummies(y['run-run'], prefix='run-run'), left_index=True, right_index=True)
del y['run-run']
del data['run-run']
X = data.astype('float64')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

SC = StandardScaler()
SC.fit(X_train)
X_train_std = np.array(SC.transform(X_train))
X_test_std = np.array(SC.transform(X_test))
y_train = np.array(y_train).reshape((-1,3))
y_test = np.array(y_test).reshape((-1,3))

class ReLu:
    def __init__(self):
        self.mask = None
    def forward(self,x):
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask] = 0
        return out
    def backward(self,dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class sigmoid:
    def __init__(self):
        self.out = None
    def forward(self,x):
        out = 1/(1+np.exp(-x))
        self.out = out
        return out
    def backward(self,dout):
        dx = dout*(1.0-self.out)*self.out
        return dx
    
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    def forward(self,x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out
    def backward(self,dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T,dout)
        self.db = np.sum(dout, axis = 0)
        return dx

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y+1e-7))/batch_size

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t =None
    def forward(self,x,t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        dx = (self.y-self.t)/batch_size
        return dx

class Norm:
    def __init__(self, r = 1, c = 0):
        self.r = r
        self.c = c
        self.dr = None
        self.dc = None
    def forward(self, x):
        self.x = x
        self.s = np.sum(self.x, axis = 1)/self.x.shape[1]
        self.sq = np.sum((self.x.T - self.s)**2, axis = 0)/ self.x.shape[1]
        self.out = (self.x.T - self.s)/np.sqrt(self.sq + 1e-7)
        return self.out.T
    def backward(self, dout):
        m = dout.shape[1]
        delta = np.sum((self.x.T - self.s)*dout.T, axis = 0)
        self.dr = np.sum(self.out.T*dout, axis = 1)
        self.dc = np.sum(dout, axis = 0)
        dx = self.r/np.sqrt(self.sq + 1e-7) + self.r/np.sqrt(self.sq + 1e-7)*np.sum(dout.T, axis = 0) - self.r*delta*np.sqrt(self.sq + 1e-7)*(self.x.T - self.s)/m
        return dx.T
    
def numerical_gradient(f,x):
    h = 1e-7
    grad = np.zeros_like(x.size)
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val -h
        fxh2 = f(x)
        grad[idx] = (fxh1-fxh2)/(2*h)
        x[idx] = tmp_val
    return grad

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.1):
        self.params = {}
        self.params['W1'] = weight_init_std*np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['Nr1'] = 1
        self.params['Nc1'] = 0
        self.params['W2'] = weight_init_std*np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        self.layers = collections.OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Norm1'] = Norm(self.params['Nr1'], self.params['Nc1'])
        self.layers['ReLu1'] = sigmoid()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()
    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    def loss(self,x,t):
        y = self.predict(x)
        return self.lastLayer.forward(y,t)
    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y, axis =1)
        if t.ndim != 1:
            t = np.argmax(t, axis =1)
        accuracy = np.sum(y == t)/float(x.shape[0])
        return accuracy
    def numerical(self,x,t):
        loss_W = lambda W:self.loss(x,t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads
    def gradient(self,x,t):
        self.loss(x,t)
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['Nr1'] = self.layers['Norm1'].dr
        grads['Nc1'] = self.layers['Norm1'].dc
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads
    
network = TwoLayerNet(input_size = 12, hidden_size = 100, output_size = 3)
learning_rate = 0.01
train_loss_list = []
train_acc_list = []
test_acc_list = []

for i in range(10000):
    
    grad = network.gradient(X_train_std, y_train)
    for key in ('W1', 'b1', 'Nr1', 'Nc1', 'W2', 'b2'):
        network.params[key] -= learning_rate*grad[key]
    loss = network.loss(X_train_std, y_train)
    train_loss_list.append(loss)
    train_acc = network.accuracy(X_train_std, y_train)
    test_acc = network.accuracy(X_test_std, y_test)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    msg ='\r진행률%d%%'%(int(i)+1)
    print(' '*len(msg),end='')
    print(msg,end='')