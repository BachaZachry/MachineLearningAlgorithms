import numpy as np
import matplotlib.pyplot as plt

#plotting the data.
def plot_data(X,y):
    plt.scatter(X,y,color="red")
    plt.show()

def compute_cost_function(X,y,theta):
    m = len(y)
    hypothesis = X.dot(theta) # calculating the hypothesis h(x)=theta0 + theta1 * x
    diff = np.subtract(hypothesis,y)
    diff_squared = np.square(diff)
    sum= np.sum(diff_squared)
    J = sum / (2 * m) # Calculating cost fonction J = (1/2m) * sum((h(x)-y)^2)
    return J
def gradient_descent(X,y,theta,alpha,iterations):
    m = len(y)
    for i in range(iterations):
        hypothesis = X.dot(theta)
        diff = np.subtract(hypothesis,y)
        X_transpose = np.transpose(X)
        derivitive_times_alpha = (alpha/m) * X_transpose.dot(diff) #Calculating the partial derivative * alpha
        theta = np.subtract(theta,derivitive_times_alpha) # Calculating theta simultaneously using matrix operations
    return theta

def run():
    #extracting data from our dataset
    dataset = np.genfromtxt('one_variable_gd_data.txt',delimiter=',')
    X = dataset[:,0] #X values
    y = dataset[:,1] #y values
    plot_data(X,y)
    b = np.ones((len(X),2)) #Creating a len(X) x 2 matrix full of 1's
    b[:, 1] = X #Keeping the first column filled with 1's to correspond to theta_zero and changing the second column
    X = b #Reassigning
    theta = [0 ,0] #Initializing random theta value
    J = compute_cost_function(X,y,theta)
    alpha = 0.01
    iterations = 1500
    theta = gradient_descent(X,y,theta,alpha,iterations)
    print("Theta value that minimize the cost function J is: {}".format(theta))

if __name__ == '__main__':
    run()