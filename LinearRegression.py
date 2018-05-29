import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalize(column):
    return (column - column.min()) / (column.max() - column.min())

filename = "dataset.xls"

dataset = pd.read_excel(filename)

dataset.columns = ['chirps/sec', 'temperature']

dataset['chirps/sec'] = normalize(dataset['chirps/sec']) 
dataset['temperature'] = normalize(dataset['temperature']) 

datasetTrain = dataset.sample(frac = 0.75)

xTrain = np.array(datasetTrain.drop('temperature', axis = 1))
xTrain = np.concatenate([np.ones([xTrain.size, 1]), xTrain], axis = 1)

yTrain = np.array(datasetTrain.drop('chirps/sec', axis = 1))
       
theta = np.zeros([1, 2])

m = yTrain.size    

alpha = 0.9

costArray = []

noOfIterations = 10000

for i in range(noOfIterations):
    
    h = np.dot(xTrain, theta.T)
    
    J = np.dot((h - yTrain).T, (h - yTrain))
    
    costArray += [J[0]]
    
    theta = theta - np.dot((h - yTrain).T, xTrain) * alpha / m
                          
                          
datasetTest = dataset.sample(frac = 0.4)

xTest = np.array(datasetTest.drop('temperature', axis = 1))
xTest = np.concatenate([np.ones([xTest.size, 1]), xTest], axis = 1)
 
yTest = np.array(datasetTest.drop('chirps/sec', axis = 1))

yPred = np.dot(xTest, theta.T)

plt.subplot(331)

plt.scatter(xTest[:, 1], yPred) 

plt.subplot(333)

plt.scatter(xTest[:, 1], yTest)

plt.subplot(337)
                                                                      
plt.plot(range(noOfIterations), costArray)        



