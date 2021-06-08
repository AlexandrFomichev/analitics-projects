# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 09:44:06 2021

@author: fomichev.aleksandr
"""

import prepareData as prep
from sklearn import preprocessing as pp
from sklearn import model_selection as ms
import pandas as pd
import numpy as np
import neuralNetwork as nw
import matplotlib.pyplot as plt

dataset=pd.read_excel('dataForecast.xlsx')

prep.cleaning.strToIndex([dataset['Region'], dataset['RetailStoreClass']])


setForModel=dataset[['Region', 'Weeks gone', 'firstStore', 'TradeArea', 'RetailStoreClass', 'Выручка РЦ']]
coeffList=list()
for i in setForModel:
    setForModel[i], coeff=prep.cleaning.normalization(setForModel[i], 'l1')[0], prep.cleaning.normalization(setForModel[i], 'l1')[1]
    coeffList.append([i, coeff])
    pass
print(coeffList)
setForModel.info()

setForModel=prep.cleaning.kvantil(setForModel, ['Выручка РЦ'], '97%', '3%')
setForModel.corr()

trainSet, testSet=ms.train_test_split(setForModel, train_size=0.78)
x_train=np.array(trainSet[['Region', 'Weeks gone', 'firstStore', 'TradeArea', 'RetailStoreClass']])
y_train=np.array(trainSet['Выручка РЦ'])
#%%

n1=nw.neuralNetwork(5, 2, 12, 1, 0.15)
n2=nw.neuralNetwork(5, 1, 12, 1, 0.15)
n3=nw.neuralNetwork(5, 3, 12, 1, 0.15)

dev1=list()
dev2=list()
dev3=list()
num_of_train=list()
#%%
for i in range(250, 1000):
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


