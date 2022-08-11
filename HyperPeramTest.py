# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 09:48:58 2022

@author: johnn

To do: 
1. Implement numLayers into createNN
2. Generate CSV for wated eval metrics
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import DigitalTwinDataGenerator as gen
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics as sk
import modelGenerator.py as modelGen

def createNN(x_train, y_train, iterations, numLyaers, numNodes, opt):
    xNorm = np.array(x_train)
    normLayer = keras.layers.Normalization(input_shape=[2,], axis=None)
    normLayer.adapt(xNorm) 
    #defines number of nodes for the hidden layers
    #Create model for NN
    model = tf.keras.Sequential([
      normLayer,
      tf.keras.layers.Dense(numNodes, activation = 'relu'),
      tf.keras.layers.Dense(numNodes, activation = 'relu'),
      tf.keras.layers.Dense(numNodes, activation = 'relu'), 
      tf.keras.layers.Dense(numNodes, activation = 'relu'), 
      tf.keras.layers.Dense(numNodes, activation = 'relu'),
      tf.keras.layers.Dense(1)])
    
    model.compile(loss='mean_absolute_error', optimizer= opt)
    fit = model.fit(x_train, y_train, batch_size = 128, epochs = iterations, verbose = 0)
    return model

def removeRange(df1, df2, low, high):
    indecies = df1[(df1['k'] > low) & (df1['k'] < high)].index
    df1.drop(indecies, inplace=True)
    df2.drop(indecies, inplace=True)
        
def keepRange(df1, df2, low, high):
    indecies = df1[(df1['k'] < low) | (df1['k'] > high)].index
    df1.drop(indecies, inplace=True)
    df2.drop(indecies, inplace=True)
       

def createPlot(x_train, y_train, x_test, yPred, name):
    plt.figure(name)
    plt.scatter(x_train, y_train, s=5, label = 'Training Data')
    plt.scatter(x_test, yPred, s=5, label='Test Predictions')
    plt.xlabel('k')
    plt.ylabel('x(t)')
    plt.title(name)
    plt.legend()
    plt.grid(True)

def testHyperParams(numLayers, numNodes, epochs, numTests, kLower, kUpper):
    dataMin = []
    dataMax = []
    dataMean = []
    dataMedian = []
    
    errorMin = []
    errorMax = []
    errorMean = []
    errorMedian = []
    mse = []
    
    dataGenerator = gen.DataGen()
    for i in range (0, numTests):
        #Use gen to generate datset.
        data = dataGenerator.generate(1000, kLower, kUpper)
                
        #Create x and y inputs for NN
        x = pd.DataFrame({'k':data.k, 'x_0':data.x_zero})
        y = data.x_t
        
            
        #Split x and y into train and test
        x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)
        test = x_train[x_train['k'] > 2]
        
        model = createNN(x_train, y_train, epochs, numLayers, numNodes, keras.optimizers.Adam(learning_rate = .01))
        yPred = model.predict(x_test)
        error = abs(np.array(yPred[:,0]) - np.array(y_test))
        
        dataMin.append((min(np.array(data.k))))
        dataMax.append((max(np.array(data.k))))
        dataMean.append((np.mean(np.array(data.k))))
        dataMedian.append((np.median(np.array(data.k))))
        
        errorMin.append(min(error))
        errorMax.append(max(error))
        errorMean.append(np.mean(error))
        errorMedian.append(np.median(error))
        
        mse.append(sk.mean_squared_error(y_test, yPred))
        
    result = pd.DataFrame({"dataMax":dataMax, "dataMean":dataMean,"dataMin":dataMin, "dataMedian":dataMedian, 
                           "errorMax":errorMax, "errorMean":errorMean,"errorMin":errorMin, "errorMedian":errorMedian,
                           "Mse":mse})
    return result

def testHyperParamsHoldOut(numLayers, numNodes, epochs, numTests, kLower, kUpper, hLower, hUpper):
    dataMin = []
    dataMax = []
    dataMean = []
    dataMedian = []
    
    errorMin = []
    errorMax = []
    errorMean = []
    errorMedian = []
    
    mse = []
    
    
    dataGenerator = gen.DataGen()
    for i in range (0, numTests):
        #Use gen to generate datset.
        data = dataGenerator.generate(1000, kLower, kUpper)
                
        #Create x and y inputs for NN
        x = pd.DataFrame({'k':data.k, 'x_0':data.x_zero})
        y = data.x_t
        
            
        #Split x and y into train and test
        x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42)
        removeRange(x_train, y_train, hLower, hUpper)
        keepRange(x_test, y_test, hLower, hUpper)
        
        model = createNN(x_train, y_train, epochs, numLayers, numNodes, keras.optimizers.Adam())
        yPred = model.predict(x_test)
        error = abs(np.array(yPred[:,0]) - np.array(y_test))
        createPlot(x_train.k, y_train, x_test.k, yPred, "Neural Network for 3 < k < 3.6")
        dataMin.append((min(np.array(data.x_t))))
        dataMax.append((max(np.array(data.x_t))))
        dataMean.append((np.mean(np.array(data.x_t))))
        dataMedian.append((np.median(np.array(data.x_t))))
        
        errorMin.append(min(error))
        errorMax.append(max(error))
        errorMean.append(np.mean(error))
        errorMedian.append(np.median(error))
        
        mse.append(sk.mean_squared_error(y_test, yPred))
    
    result = pd.DataFrame({"dataMax":dataMax, "dataMean":dataMean,"dataMin":dataMin, "dataMedian":dataMedian, 
                           "errorMax":errorMax, "errorMean":errorMean,"errorMin":errorMin, "errorMedian":errorMedian,
                           "Mse":mse})
    return result


result1 = modelGen.testHyperParams(5, 128, 1000, 20, 0, 3)
result2 = modelGen.testHyperParamsHoldOut(5, 128, 1000, 20, 0, 3, 1.2, 1.3) 
result3 = modelGen.testHyperParams(5, 128, 1000, 20, 3, 3.6)
result4 = modelGen.testHyperParamsHoldOut(5, 128, 1000, 20, 3, 3.6, 3.3, 3.35)
result5 = modelGen.testHyperParams(5, 128, 1000, 20, 3.6, 4)  
result6 = modelGen.testHyperParamsHoldOut(5, 128, 1000, 20, 3.6, 4, 3.7, 3.75)    



