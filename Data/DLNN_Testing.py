# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 17:57:57 2022

@author: johnn
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import DigitalTwinDataGenerator as gen
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics as sk

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
    model.fit(x_train, y_train, epochs = iterations, verbose = 0)
    return model

def testHyperParams(numLayers, numNodes, epochs, numTests, kLower, kUpper, lr):
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
        
        model = createNN(x_train, y_train, epochs, numLayers, numNodes, keras.optimizers.Adam(learning_rate = lr))
        yPred = model.predict(x_test)
        error = abs(np.array(yPred[:,0]) - np.array(y_test))
        
        #Calculate data stats
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

result = testHyperParams(5, 128, 100, 5, 0, 4, .001)