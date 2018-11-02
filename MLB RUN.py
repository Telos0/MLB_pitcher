# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 12:17:21 2018

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

data['run_to_run'] = abs(data['away_team_runs']+data['home_team_runs'])
del data['away_team_runs']
del data['home_team_runs']
del data['total_runs']

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
y = data['run_to_run']
del data['run_to_run']
X = data.astype('float64')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

SC = StandardScaler()
SC.fit(X_train)
X_train_std = SC.transform(X_train)
X_test_std = SC.transform(X_test)


from sklearn.linear_model import LinearRegression, Ridge, Lasso

LR = LinearRegression(fit_intercept = True)
RI = Ridge(fit_intercept = True)
LA = Lasso(fit_intercept = True)

LR.fit(X_train_std, y_train)
RI.fit(X_train_std, y_train)
LA.fit(X_train_std, y_train)

fig, axes = plt.subplots(nrows=1,ncols=3)
fig.set_size_inches(12, 5)

axes[0].scatter(x = range(len(y_test)), y = LR.predict(X_test_std)-y_test, c = 'black')
axes[1].scatter(x = range(len(y_test)), y = LA.predict(X_test_std)-y_test, c = 'blue')
axes[2].scatter(x = range(len(y_test)), y = RI.predict(X_test_std)-y_test, c = 'red')

#axes[0].scatter(x = range(len(y_test)), y = y_test, c = 'green')
#axes[1].scatter(x = range(len(y_test)), y = y_test, c = 'green')
#axes[2].scatter(x = range(len(y_test)),y = y_test, c = 'green')

def error_for_run(X_train, y_train, X_test, y_test, clf):
    clf = clf()
    clf.fit(X_train, y_train)
    return abs(y_test - clf.predict(X_test)).mean()

'''from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, random_state = 0)
DT.fit(X_train_std, y_train)
pre = DT.predict(X_test_std)
print('DT total %d missed %d' %(len(y_test), sum(DT.predict(X_test_std) != y_test)))'''


'''fig, axes = plt.subplots(nrows=3,ncols=3)
fig.set_size_inches(12, 5)
axes[0][0].bar(data_copy["attendance"], data_copy["run_to_run"])
axes[0][1].bar(data_copy["away_team"], data_copy["run_to_run"])
axes[0][2].bar(data_copy["home_team"], data_copy["run_to_run"])
axes[1][0].bar(data_copy["venue"], data_copy["run_to_run"])
axes[1][1].bar(data_copy["day_of_week"], data_copy["run_to_run"])
axes[1][2].bar(data_copy["temperature"], data_copy["run_to_run"])
axes[2][0].bar(data_copy["wind_speed"], data_copy["run_to_run"])
axes[2][1].bar(data_copy["wind_direction"], data_copy["run_to_run"])
axes[2][2].bar(data_copy["sky"], data_copy["run_to_run"])
plt.show()'''
