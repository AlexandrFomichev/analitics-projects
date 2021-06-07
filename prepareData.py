# -*- coding: utf-8 -*-
"""
Created on Thu May  6 09:56:08 2021

@author: fomichev.aleksandr
"""
import pandas as pd
from sklearn.preprocessing import normalize as nm
import numpy as np
class cleaning:
    def __init__(self):
        
        pass
    
    def setAllTypeFloat(dataset):
        for i in dataset:
            try:
                dataset[i]=dataset[i].astype(float)
                pass
            except:
                pass
            pass
        return dataset
    
    def normalization(dataColumn, normType):
        dataColumn=np.array(dataColumn)
        normData=nm([dataColumn],norm=normType)
        if normType=='l1':
            normRate=sum(dataColumn**2)**0.5
            pass
        elif normType=='l2':
            normRate=sum(dataColumn)
            pass
        else:
            normRate=max(dataColumn)
            pass
        normData=dataColumn/normRate
        return [normData, normRate]
    
    def kvantil(dataset, fields, kvantUp, kvantDown):
        upPercent=float(kvantUp.replace('%', ''))/100
        downPercent=float(kvantDown.replace('%', ''))/100
        for i in fields:
            upValue=dict(dataset[i].describe([upPercent]))[kvantUp]
            downValue=dict(dataset[i].describe([downPercent]))[kvantDown]
            dataset=dataset[dataset[i]<=upValue][dataset[i]>=downValue]
            pass
        return dataset
        pass
    
  
    
    
    pass

    






