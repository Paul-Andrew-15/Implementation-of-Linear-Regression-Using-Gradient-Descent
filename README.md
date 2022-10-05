# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required library and read the dataframe.
2.Write a function computeCost to generate the cost function.
3.Perform iterations og gradient steps with learning rate.
4.Plot the Cost function using Gradient Descent and generate the required graph.
```
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Paul Andrew D
RegisterNumber:  212221230075
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("ex1.txt",header=None)
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
    """"
    Take in a numpy array X,y,theta and generate the cost function of using theta as a parameter in a linera regression tool   
    """
    m=len(y) 
    h=X.dot(theta) 
    square_err=(h-y)**2
    return 1/(2*m)*np.sum(square_err) 

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta) 

def gradientDescent(X,y,theta,alpha,num_iters):
    """"
    Take in numpy array X,y and theta and update theta by taking num_iters gradient steps with learning rate of alpha 
    return theta and the list of the cost of the theta during each iteration
    """
    m=len(y)
    J_history=[] #empty list
    for i in range(num_iters):
        predictions=X.dot(theta)
        error=np.dot(X.transpose(),(predictions-y))
        descent=alpha*(1/m)*error
        theta-=descent
        J_history.append(computeCost(X,y,theta))
    return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
    """"
    Takes in numpy array of x and theta and return the predicted valude of y based on theta
    """
    predictions=np.dot(theta.transpose(),x)
    return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For Population = 35000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For Population = 70000, we predict a profit of $"+str(round(predict2,0)))
```

## Output:
![193603024-336ca9fd-db24-4738-93d6-6cb1d24963b6](https://user-images.githubusercontent.com/94170892/193615212-c27f6633-616f-4220-8d24-669d6cdfce78.png)

![193603058-09715bdf-3fb8-4f85-859f-c00c24229943](https://user-images.githubusercontent.com/94170892/193615249-c2103ac4-1e89-4d05-8053-cbd657667998.png)

![193603136-85ebf796-f62c-4745-803e-a0ea08c9f62a](https://user-images.githubusercontent.com/94170892/193615277-36523c62-1600-4277-b098-cc1c54454ba2.png)

![193603208-2ddf9dd2-8b88-4d00-873c-a45d2a8e3d57](https://user-images.githubusercontent.com/94170892/193615322-566d94e8-cf1e-449b-9dbb-3ab034d4ac59.png)

![193603254-20c94237-28ee-4ecf-a23d-fc9280d9861f](https://user-images.githubusercontent.com/94170892/193615357-5e3e6328-698f-40ad-9b5f-56ce6ceb8e3a.png)

![193603300-222f70e4-a2cb-4c76-a3d4-de812f2d7fc1](https://user-images.githubusercontent.com/94170892/193615385-6430b816-9c84-4a06-a92f-07c86fb8a182.png)

![193603348-86ce9395-e480-40c7-a01a-1c3f714f9b92](https://user-images.githubusercontent.com/94170892/193615411-e50cd6ee-0267-4173-bf89-488e0d153b88.png)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
