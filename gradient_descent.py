import numpy as np
import matplotlib.pyplot as plt
def linear_regression(x,y,a_current=0,b_current=0,epoches=1000,stepsize=0.001):
    N=float(len(y))
    for i in range(epoches):
        y_current = (a_current*x)+b_current
        cost=sum([data**2 for data in (y - y_current)])/(2*N)
        a_gradient = -(1/N) * sum(x * (y - y_current))
        b_gradient = -(1/N) * sum((y - y_current))
        a_current = a_current - (stepsize * a_gradient)
        b_gradient = b_current - (stepsize * b_gradient)
    return a_current, b_current, cost

def func(x):
    return x**3 - 4.5*x**2 + 6*x + 2

x = np.random.rand(300)
tmp = np.random.rand(300)-0.5
x=x*3
print(tmp)
x.sort()
y=func(x)+tmp
print(x)
print(y)

plt.plot(x,y,'bo')
plt.show()

cx,cy,cost=linear_regression(x,y)
plt.plot(cx,cy,'bo')