# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 17:16:54 2018

@author: Mohak
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

trainingSet = pd.read_csv('G:\Google_Stock_Price_Train.csv')
trainingSet = trainingSet.iloc[:,1:2].values

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
trainingSet = sc.fit_transform(trainingSet)

#Getting input and output
xTrain = trainingSet[0:1257]
yTrain = trainingSet[1:1258]

#Reshaping the inputs
xTrain = np.reshape(xTrain, (1257, 1, 1 ))

#Importing the packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#Initializing the RNN
model = Sequential()
#this is a regression model because the output would be a real number and not 0 or 1
#Adding the LSTM layer
model.add(LSTM(units = 4, activation = 'sigmoid', input_shape=(None, 1)))
#Adding the output layer
#we'll keep most of the things as default
#units corrospond to the no  of neurons in the output layer, here the output is 1D (units=1)
model.add(Dense(units=1))
#Compiling the RNN
model.compile(optimizer='adam', loss = 'mean_squared_error')
#Fitting the model
model.fit(x =xTrain , y =yTrain , batch_size=32, epochs=200)
model.save_weights('G:\TrainedRNN.h5')

#Test Set
testSet = pd.read_csv('G:\Google_Stock_Price_Test.csv')
realStockPrice = testSet.iloc[:, 1:2].values
inputs = realStockPrice
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (20,1,1))
prediction = model.predict(inputs)

#now the predicted output is scaled, we will apply the reverse transform method to get the actual predicted prices
prediction = sc.inverse_transform(prediction)

#Visualization the results
plt.plot(realStockPrice, color = 'Red', label='RealPrice')
plt.plot(prediction, color = 'Blue', label='Predicted')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Google Stock Price')
plt.legend()
plt.show()
