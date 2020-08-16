# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:06:19 2019

@author: \devi karthika p
"""

#importing the necessary python packages
#numpy to calculate the mean square error
#pandas to reaad the csv file
#matplotlib to plot the graph
#sklearn to implement polynomial regression
import numpy as np 
import pandas as pd
#%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error 

#Read the file Book1 which is our dataset
dataset = pd.read_csv("C:\\Users\\devikarthikap\\Documents\\Book1.csv")
#selecting the features from the dataset
X = dataset.drop("Dynamic_CP", axis=1)
#selecting the target from the dataset
y = dataset["Dynamic_CP"]
#classifying the dataset into test data and training data and setting the parameters
x_training_set, x_test_set, y_training_set, y_test_set = train_test_split(X,y,test_size=0.10, 
                                                                          random_state=8,
                                                                          shuffle=True)
# Polynomial Regression-7th order
for degree in [7]:
    #making a pipeline with polynomial regression of 7th order and linear regression
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    
    #fitting the training data
    model.fit(x_training_set,y_training_set)

    #predicting for test data
    y_plot = model.predict(x_test_set)
    #predicting for training data
    y1_plot= model.predict(x_training_set)
    

    #printing the predicted values for test data
    print(y_plot)
    #printing the test target for comparison
    print(y_test_set)
    
    #calculating the mean square error of test data
    test_set_rmse = (np.sqrt(mean_squared_error(y_test_set,y_plot)))
    #calculating the mean square error of training data
    train_set_rmse=(np.sqrt(mean_squared_error(y_training_set,y1_plot)))
    
    #printing the test data MSE
    print(test_set_rmse)
    #printing the training data MSE
    print(train_set_rmse)
    
    #plotting the graph of varaince explained with the degree of the polynomial
    plt.plot(x_test_set, y_plot, label="degree %d" % degree

             +'; $R^2$: %.2f' % model.score(x_test_set, y_test_set))

plt.legend(loc='upper right')
#labelling the x-axis as the test data
plt.xlabel("Test  Data")
#labelling the y-axis as training data
plt.ylabel("Predicted Price")
#title of the graph
plt.title("Variance Explained with 7th degree Polynomial")
plt.show()
