import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('Position_Salaries.csv')


dataset
x=dataset.iloc[:,1:2].values

y=dataset.iloc[:,2:].values
x


fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.scatter(x,y,color='r')



from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly=PolynomialFeatures(degree=2)
x_poly=poly.fit_transform(x)

pilreg=LinearRegression()
pilreg.fit(x_poly,y)

plt.scatter(x,y,color='red')
plt.plot(x,pilreg.predict(poly.fit_transform(x)),color='blue')


pilreg.predict(poly.fit_transform([[10]]))