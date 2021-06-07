# -*- coding: utf-8 -*-
"""
Created on Fri May 14 11:02:22 2021

@author: fomichev.aleksandr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing as pp
from sklearn import model_selection as ms
import math as mp
from prepareData import cleaning as cl


dataset=pd.read_excel("store_data_2.xlsx")
dataset.drop(columns=["store", "yearOfSales"], inplace=True)


dataset=cl.kvantil(dataset, ['Revenue'], '95%', '5%')

trainSet, testSet=ms.train_test_split(dataset, train_size=0.78)

trainSet.info()
testSet.info()

