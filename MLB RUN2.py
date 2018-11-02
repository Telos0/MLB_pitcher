# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 16:52:45 2018

@author: telos

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    if 1<=x<=4:
        return 2
    elif 4<x<=7:
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

del data['away_team']

#copy
data_copy = data.copy()

#start
y = data['run-run']
del data['run-run']
X = data.astype('float64')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#SC = StandardScaler()
#SC.fit(X_train)
#X_train_std = SC.transform(X_train)
#X_test_std = SC.transform(X_test)
X_train_std = X_train
X_test_std = X_test

from sklearn.linear_model import LogisticRegression

LG = LogisticRegression()
LG.fit(X_train_std, y_train)
print('LogisticRegression total %d missed %d' %(len(y_test), sum(LG.predict(X_test_std) != y_test)))
score = LG.score(X_test_std, y_test)
score = float(score)*100
print('Logistic Regression acc : %f%%'% score)

from sklearn.naive_bayes import GaussianNB

GNB = GaussianNB()
GNB.fit(X_train_std, y_train)
print('GaussianNB total %d missed %d' %(len(y_test), sum(GNB.predict(X_test_std) != y_test)))
score = float((len(y_test)-sum(GNB.predict(X_test_std) != y_test))/len(y_test))*100
print('GaussianNB acc : %f%%' % score)

from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3)
DT.fit(X_train_std, y_train)
print('DecisionTree total %d missed %d' %(len(y_test), sum(DT.predict(X_test_std) != y_test)))
score = float((len(y_test)-sum(DT.predict(X_test_std) != y_test))/len(y_test))*100
print('DecisionTree acc : %f%%' % score)

from sklearn.linear_model import SGDClassifier

SGD = SGDClassifier(eta0 = 0.0001, learning_rate = 'invscaling',loss = 'hinge', penalty = 'l1', max_iter = 10000, random_state = 0)
SGD.fit(X_train_std, y_train)
a = SGD.predict(X_test_std)
print('SGD total %d missed %d' %(len(y_test), sum(a != y_test)))
score = float((len(y_test)-sum(a != y_test))/len(y_test))*100
print('SGD acc : %f%%' % score)


    
