import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
class maby:
    def __init__(self):
        self.a=float(np.random.randint(1,5))
        self.b=float(np.random.randint(1,5))
        self.c=float(np.random.randint(1,5))
        self.d=float(np.random.randint(1,5))
        self.e=float(np.random.randint(1,5))
        self.f=float(np.random.randint(1,5))
    def func(self, x):
        return self.a*(x**5)+self.b*(x**4)+self.c*(x**3) + self.d*(x**2) + self.e*x + self.f
    def show(self):
        print("x**3 - 4.5*x**2 + 6*x + 2")
        print(format(self.a,'.1f')+"x**3 -"+format(self.a,'.1f')+"*x**2 + "+format(self.a,'.1f')+"*x + "+format(self.a,'.1f'))

def inf_linear_regression(x, y, expect ,epoches=1000,stepsize=0.01):
    N=float(len(x))
    plt.plot(x,y,'r-',markersize=2,label="goal")
    plt.plot(x,expect.func(x),'g-',markersize=2,label="expect")
    count=0
    postcost=0
    while True:
        y_current = expect.func(x)
        cost=np.sum((y - y_current)**2)/(2 * N)
        if cost > postcost:
            count +=1
        elif cost==postcost:
            print(a_gradient,b_gradient,c_gradient,d_gradient,stepsize)
            break
        if count>10:
            count=0
            if stepsize< 0.0001:
                stepsize=stepsize
            else:
                stepsize=stepsize/10
        a_gradient = -(1/N) * np.sum(x**5 * (y - y_current))
        b_gradient = -(1/N) * np.sum(x**4 * (y - y_current))
        c_gradient = -(1/N) * np.sum(x**3 * (y - y_current))
        d_gradient = -(1/N) * np.sum(x**2 * (y - y_current))
        e_gradient = -(1/N) * np.sum(x * (y - y_current))
        f_gradient = -(1/N) * np.sum((y - y_current))
        expect.a -= (stepsize * a_gradient)
        expect.b -= (stepsize * b_gradient)
        expect.c -= (stepsize * c_gradient)
        expect.d -= (stepsize * d_gradient)
        expect.e -= (stepsize * e_gradient)
        expect.f -= (stepsize * f_gradient)
        print(cost)
        plt.plot(x,func(x),'r-',markersize=2,label="goal")
        plt.plot(x,expect.func(x),'g-',markersize=2,label="expect")
        plt.legend()
        plt.draw()
        plt.pause(0.000001)
        plt.cla()
        postcost=cost
    return cost

def linear_regression(x, y, expect ,epoches=1000,stepsize=0.001):
    N=float(len(x))
    plt.plot(x,y,'r-',markersize=2,label="goal")
    plt.plot(x,expect.func(x),'g-',markersize=2,label="expect")
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
        print(cost)
        plt.plot(x,func(x),'r-',markersize=2,label="goal")
        plt.plot(x,expect.func(x),'g-',markersize=2,label="expect")
        plt.legend()
        plt.draw()
        plt.pause(0.000001)
        plt.cla()
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

cost=inf_linear_regression(x,y2,expect,epoches=1, stepsize=0.001)
expect.show()
cy1=expect.func(x)
plt.plot(x,y1,'r-',markersize=2,label="goal")
plt.plot(x,cy1,'g-',markersize=2,label="expect")
plt.legend()
plt.show()
# cost=linear_regression(x,y2,expect,epoches=10, stepsize=0.001)
# expect.show()
# cy2=expect.func(x)
# plt.plot(x,y1,'r-',markersize=2,label="goal")
# plt.plot(x,cy1,'g-',markersize=2,label="expect1")
# plt.plot(x,cy2,'b-',markersize=2,label="expect10")
# plt.legend()
# plt.show()
# cost=linear_regression(x,y2,expect,epoches=100,stepsize=0.001)
# expect.show()
# cy3=expect.func(x)
# plt.plot(x,y1,color="red",markersize=2,label="goal")
# plt.plot(x,cy1,color="green",markersize=2,label="expect1")
# plt.plot(x,cy2,color="blue",markersize=2,label="expect10")
# plt.plot(x,cy3,color="violet",markersize=2,label="expect100")
# plt.legend()
# plt.show()
# cost=linear_regression(x,y2,expect,epoches=1000,stepsize=0.001)
# expect.show()
# cy3=expect.func(x)
# plt.plot(x,y1,color="red",markersize=2,label="goal")
# plt.plot(x,cy3,color="violet",markersize=2,label="expect100")
# plt.legend()
# plt.show()
