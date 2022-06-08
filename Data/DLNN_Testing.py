"""
Created on Mon Jun  6 14:19:28 2022
@author: johnn
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import DigitalTwinDataGenerator as gen
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

#Create DigitalTwinDataGenerator object
gen = gen.DataGen()  

#Use gen to generate datset 
data = gen.generate(1000, 0, 2.9)

#Create x and y inputs for NN
x = pd.DataFrame(data.k)
x.insert(1, "x_zero", data.x_zero, True)

y = pd.DataFrame(data.x_t)
y.insert(1, "Fatigue", data.Fatigue, True)

#Split x and y into train and test
x_train,x_test, y_train, y_test, = train_test_split(x,y,test_size=0.2, random_state=42)
xNorm = np.array(x_train)
normLayer = keras.layers.Normalization(input_shape=[2,], axis=None)
normLayer.adapt(xNorm)

#defines number of nodes for the hidden layers
numNodes = 32

#Create model for NN
model = tf.keras.Sequential([
  normLayer,
  tf.keras.layers.Dense(numNodes, activation = 'relu'),
  tf.keras.layers.Dense(numNodes, activation = 'relu'),
  tf.keras.layers.Dense(2)])

model.compile(loss='mean_absolute_error', optimizer=keras.optimizers.Adam(0.001))
fit = model.fit(x_train, y_train, epochs = 10)

#create graphs for the testing performance of x(t) and Fatigue
yPred = model.predict(x_test)
figure, axis = plt.subplots(2,1)


axis[0].scatter(x_test["k"], y_test["x_t"], label='Data')
axis[0].scatter(x_test["k"], yPred[0:,0], color='k', label='Prediction')
axis[0].set_title("DLNN for K Less Than Three")
axis[0].legend()
axis[0].grid(True)

axis[1].scatter(x_test["k"], y_test["Fatigue"], label='Data')
axis[1].scatter(x_test["k"], yPred[0:,1], color='k', label='Prediction')
axis[1].legend()
axis[1].grid(True)

plt.show()
