import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
class maby:
    def __init__(self):
        self.a=float(np.random.randint(1,5))
        self.b=float(np.random.randint(1,5))
        self.c=float(np.random.randint(1,5))
        self.d=float(np.random.randint(1,5))
    def func(self, x):
        return self.a*(x**3) - self.b*(x**2) + self.c*x + self.d
    def show(self):
        print("x**3 - 4.5*x**2 + 6*x + 2")
        print(format(self.a,'.1f')+"x**3 -"+format(self.a,'.1f')+"*x**2 + "+format(self.a,'.1f')+"*x + "+format(self.a,'.1f'))

def linear_regression(x, y, expect ,epoches=1000,stepsize=0.001):
    N=float(len(x))
    for i in range(epoches):
        y_current = expect.func(x)
        cost=np.sum((y - y_current)**2)/(2 * N)
        a_gradient = -(1/N) * np.sum(x**3 * (y - y_current))
        b_gradient = -(1/N) * np.sum(x**2 * (y - y_current))
        c_gradient = -(1/N) * np.sum(x * (y - y_current))
        d_gradient = -(1/N) * np.sum((y - y_current))
        expect.a -= (stepsize * a_gradient)
        expect.b -= (stepsize * b_gradient)
        expect.c -= (stepsize * c_gradient)
        expect.d -= (stepsize * d_gradient)
    return cost

def func(x):
    return x**3 - 4.5*x**2 + 6*x + 2
expect=maby()
x = np.random.rand(300)
tmp = np.random.rand(300)-0.5
x=x*3
x.sort()
y1=func(x)
y2=func(x)+tmp
plt.plot(x,y1,'r-',markersize=2,label="goal")
plt.plot(x,y2,'bo',markersize=2,label="noise")
plt.legend()
plt.show()

cost=linear_regression(x,y2,expect,epoches=100000, stepsize=0.0001)
expect.show()
cy1=expect.func(x)
plt.plot(x,y1,'r-',markersize=2,label="goal")
plt.plot(x,cy1,'g-',markersize=2,label="expect")
plt.legend()
plt.show()
cost=linear_regression(x,y2,expect,epoches=100000, stepsize=0.0001)
expect.show()
cy2=expect.func(x)
plt.plot(x,y1,'r-',markersize=2,label="goal")
plt.plot(x,cy1,'g-',markersize=2,label="expect")
plt.plot(x,cy2,'b-',markersize=2,label="expect")
plt.legend()
plt.show()
cost=linear_regression(x,y2,expect,epoches=100000,stepsize=0.0001)
expect.show()
cy3=expect.func(x)
plt.plot(x,y1,color="red",markersize=2,label="goal")
plt.plot(x,cy1,color="green",markersize=2,label="expect")
plt.plot(x,cy2,color="blue",markersize=2,label="expect")
plt.plot(x,cy3,color="violet",markersize=2,label="expect")
plt.legend()
plt.show()

