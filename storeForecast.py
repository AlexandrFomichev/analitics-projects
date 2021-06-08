# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 09:44:06 2021

@author: fomichev.aleksandr
"""

import prepareData as prep
from sklearn import model_selection as ms
import pandas as pd
import numpy as np
import neuralNetwork as nw
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as reg
from sklearn.ensemble import RandomForestRegressor as forest
from sklearn.metrics import r2_score as r2
from sklearn.neighbors import KNeighborsRegressor as neighbor

dataset=pd.read_excel('dataForecast.xlsx')

prep.cleaning.strToIndex([dataset['Region'], dataset['RetailStoreClass']])



coeffList=list()
for i in ['Region', 'Weeks gone', 'firstStore', 'TradeArea', 'RetailStoreClass', 'Выручка РЦ']:
    dataset[i], coeff=prep.cleaning.normalization(dataset[i], 'l1')[0], prep.cleaning.normalization(dataset[i], 'l1')[1]
    coeffList.append([i, coeff])
    pass

coeffList=dict(coeffList)

dataset=prep.cleaning.kvantil(dataset, ['Выручка РЦ'], '95%', '5%')
dataset.corr()

trainSet, testSet=ms.train_test_split(dataset, train_size=0.6)
x_train=np.array(trainSet[['Region', 'Weeks gone', 'firstStore', 'TradeArea', 'RetailStoreClass']])
y_train=np.array(trainSet['Выручка РЦ'])

x_test=np.array(testSet[['Region', 'Weeks gone', 'firstStore', 'TradeArea', 'RetailStoreClass']])
y_test=np.array(testSet['Выручка РЦ'])
#%%

n1=nw.neuralNetwork(5, 1, 12, 1, 0.1)
n2=nw.neuralNetwork(5, 1, 25, 1, 0.1)
n3=nw.neuralNetwork(5, 2, 12, 1, 0.1)


dev1=list()
dev2=list()
dev3=list()
num_of_train=list()
#%%
##n1.learning_rate=0.32
for i in range(100, 10000):
    n1.train_n(x_train, y_train,5)
    n2.train_n(x_train, y_train,5)
    n3.train_n(x_train, y_train,5)
    dev1.append(n1.showDeviation(x_train,y_train))
    dev2.append(n2.showDeviation(x_train,y_train))
    dev3.append(n3.showDeviation(x_train,y_train))
    num_of_train.append(i)
    pass
#%%
plt.plot(num_of_train,dev1)
plt.plot(num_of_train,dev2)
plt.plot(num_of_train,dev3)
plt.legend(["n1", "n2", "n3"], loc ="lower right")
plt.show()
#%%
valid=dataset[dataset['Наименование склада']=='312 Вэйпарк Москва']
X=np.array(valid[['Region', 'Weeks gone', 'firstStore', 'TradeArea', 'RetailStoreClass']])
Y=np.array(valid[['Выручка РЦ']])
param=np.array(valid['Weeks gone'])
n1.show2D(X, Y, param)
#%%

n1.showDeviation(x_test,y_test)

tree=forest(n_estimators=300, max_features='sqrt')
tree.fit(x_train, y_train)
r2(y_test, tree.predict(x_test))

nei=neighbor(n_neighbors=5)
nei.fit(x_train, y_train)
r2(y_test, nei.predict(x_test))

lReg=reg()
lReg.fit(x_train, y_train)
r2(y_test, lReg.predict(x_test))
