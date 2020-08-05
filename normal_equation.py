import numpy as np
import pandas as pd

#extracting data from csv file
dataset = pd.read_csv('Real_estate_dataset.csv')
dataset = dataset.drop(columns="No")
print(dataset.head())
#forming x,y matrices
x = dataset.iloc[:, :-1].values #x is a matrix of size 414 x 6
y = dataset.iloc[:,-1].values #y is a vector of size 414
#reforming x matrix to fit hypothesis

b = np.ones((x.shape[0],x.shape[1]+1)) #b is a matrix of size 414 x 7 filled with ones
b[:,1:] = x # keeping the first column filled with ones and changing the other columns to x matrix values
x = b #reassigning

#Applying normal equation
transpose_x = np.transpose(x)
x_transpose_times_x = transpose_x.dot(x)
x_transpose_times_x_inverse = np.linalg.inv(x_transpose_times_x)
x_transpose_times_x_inverse_times_x_transpose = x_transpose_times_x_inverse.dot(transpose_x)
#result
theta = x_transpose_times_x_inverse_times_x_transpose.dot(y)
print(theta)
#predicting
test_x = [1,2013.250,5.4,390.5684,5,24.97937,121.54245] # test_x 1 1 is always 1,text_x 1 2 : test_x 1 7 is the values for each column
transpose_theta = np.transpose(theta)
price = transpose_theta.dot(test_x) #calculating the hypothesis
print(price)