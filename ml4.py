#4. Write a python program to implement SIMPLE LINEAR REGREATION for predicting house price.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv("homeprices.csv")
x=dataset.iloc[:,0:1].values
y=dataset.iloc[:,-1].values
print(x)
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(x_test)
print(y_test)
plt.scatter(x_train,y_train,color='red',marker='*',s=100)
plt.plot(x_train,lr.predict(x_train),color='blue')
plt.title('price vs area')
plt.xlabel('area')
plt.ylabel('price')
