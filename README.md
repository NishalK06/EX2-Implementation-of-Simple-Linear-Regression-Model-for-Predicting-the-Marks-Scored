# EX2 Implementation of Simple Linear Regression Model for Predicting the Marks Scored
## AIM:
To implement simple linear regression using sklearn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y by reading the dataset.
2. Split the data into training and test data.
3. Import the linear regression and fit the model with the training data.
4. Perform the prediction on the test data.
5. Display the slop and intercept values.
6. Plot the regression line using scatterplot.
7. Calculate the MSE.

## Program:
```
/*
Program to implement univariate Linear Regression to fit a straight line using least squares.
Developed by:K.Nishal 
RegisterNumber: 2305001021 
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('/content/ex1.csv')
df.head()

plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')

from sklearn.model_selection import train_test_split
X = df['X']
y = df['Y']
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression

lr=LinearRegression()
X_train_reshaped = X_train.values.reshape(-1, 1)
lr.fit(X_train_reshaped,Y_train)

plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train, lr.predict(X_train.values.reshape(-1, 1)), color='red')

m=lr.coef_
m

b=lr.intercept_
b

pred=lr.predict(X_test.values.reshape(-1, 1))
pred

X_test

Y_test

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(Y_test, pred)
print(f"Mean Squared Error (MSE): {mse}")
```

## Output:


![image](https://github.com/user-attachments/assets/2db780f2-be8e-4e66-bb69-5263a3599282)

![image](https://github.com/user-attachments/assets/e316116d-3567-4a4b-aea0-aa894a11a2bc)

![image](https://github.com/user-attachments/assets/b4db8c09-575c-490b-a220-92128573b0b5)

![image](https://github.com/user-attachments/assets/7fcdb0db-9567-435f-bc2c-5120b2c4a0f4)

![image](https://github.com/user-attachments/assets/66e9c4d8-e531-4e09-9a24-73bc3fe7e2a4)

![image](https://github.com/user-attachments/assets/256ff9d5-dedc-4e3b-be45-0753f66125b6)

![image](https://github.com/user-attachments/assets/86c8165d-7625-49c9-8389-8f72eed16865)

![image](https://github.com/user-attachments/assets/83137a56-6840-4521-9009-2c29fa269a41)

![image](https://github.com/user-attachments/assets/400b4c21-a43e-420a-bf54-421121f7092a)

![image](https://github.com/user-attachments/assets/64a46fa5-8abe-48cc-b308-0967db5f3bc3)










## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
