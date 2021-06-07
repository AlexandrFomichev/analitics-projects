# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:31:23 2021

@author: Alexandr
"""
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score as r2

class neuralNetwork:
    def __init__(self, inputNodes, hieddenLayers, hiddenNodes, outputNodes
                 , learningRate):
        self.inodes=inputNodes
        self.hnodes=hiddenNodes
        self.onodes=outputNodes
        self.numOfHidLay=hieddenLayers
        self.lr=learningRate
        self.W=[]
        if self.numOfHidLay==0:
            self.W.append(np.random.normal(0.0, self.onodes**(-0.5), (self.onodes, self.inodes)))
            pass
        elif self.numOfHidLay==1:
            self.W.append(np.random.normal(0.0, self.hnodes**(-0.5), (self.hnodes, self.inodes)))
            self.W.append(np.random.normal(0.0, self.onodes**(-0.5), (self.onodes, self.hnodes)))
            pass
        else:
            self.W.append(np.random.normal(0.0, self.hnodes**(-0.5), (self.hnodes, self.inodes)))
            for i in range(1,self.numOfHidLay):
                self.W.append(np.random.normal(0.0, self.hnodes**(-0.5), (self.hnodes, self.hnodes)))
                pass
            self.W.append(np.random.normal(0.0, self.onodes**(-0.5), (self.onodes, self.hnodes)))
        self.activation_function=lambda x: sp.expit(x)
        pass
    

    def train(self, input_list, target_list):
        targets=np.array(target_list, ndmin=2).T
        neural_outputs=[]
        neural_outputs.append(np.array(input_list, ndmin=2).T)
        neural_inputs=(np.array(input_list, ndmin=2).T)
        for i in range(self.numOfHidLay+1):
            neural_outputs.append(self.activation_function(np.dot(self.W[i], neural_inputs)))
            neural_inputs=neural_outputs[i+1]
            pass
        
        
        output_errors=targets-neural_outputs[self.numOfHidLay+1]
        for i in range(self.numOfHidLay, -1, -1):
            self.W[i]+=self.lr*np.dot(output_errors*neural_outputs[i+1]*(1.0-neural_outputs[i+1])
                , np.transpose(neural_outputs[i]))
            output_errors=np.dot(self.W[i].T, output_errors)
            pass
        neural_outputs=[]
        neural_inputs=[]
        
        pass
    
    def train_n(self, input_list, target_list, n_trains):
        for l in range(n_trains):
            for k in range(len(input_list)):
                self.train(input_list[k], target_list[k])
                pass
            pass
        pass
    
        
    
    def showDeviation(self, input_list, target_list):
        y=target_list
        y1=[float(self.query(i)[0]) for i in input_list]
        self.sko=r2(y,y1)
        return(self.sko)
        
        

    def show2D(self, input_list, target_list, by_parameter):
        y=target_list
        y1=[float(self.query(i)[0]) for i in input_list]
        plt.scatter(by_parameter, y, c='red')
        plt.scatter(by_parameter,y1, c='blue')
        print('red - real, blue - neural')
        pass

    def query(self, input_list):
        neural_inputs=np.array(input_list, ndmin=2).T
        for i in range(self.numOfHidLay+1):
            neural_inputs=np.dot(self.W[i], neural_inputs)
            neural_outputs=self.activation_function(neural_inputs)
            neural_inputs=neural_outputs
            pass
        
        return neural_outputs
 
    def info(self):
        print('Input nodes qntt: '+str(self.inodes))
        print('Hidden nodes qntt: '+str(self.hnodes))
        print('Output nodes qntt: '+str(self.onodes))
        print('Layers qntt: '+str(self.numOfHidLay))
        print('Count of W-arrays: '+str(len(self.W)))
        for i in range(self.numOfHidLay+1):
            print(self.W[i])
            print('     ')
            pass
        pass
    
        

    pass



    


    

    

