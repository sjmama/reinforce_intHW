import numpy as np
import matplotlib.pyplot as plt
def func(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def gradient_descent(x, y, learning_rate=0.01, epochs=1000):
    a, b, c, d = np.random.randn(4) # Random initialization of parameters
    N = len(x)
    for _ in range(epochs):
        y_current = func(x, a, b, c, d)
        a_gradient = -(2/N) * np.sum(x**3 * (y - y_current))
        b_gradient = -(2/N) * np.sum(x**2 * (y - y_current))
        c_gradient = -(2/N) * np.sum(x * (y - y_current))
        d_gradient = -(2/N) * np.sum(y - y_current)
        a -= learning_rate * a_gradient
        b -= learning_rate * b_gradient
        c -= learning_rate * c_gradient
        d -= learning_rate * d_gradient
    return a, b, c, d

# Generating data
np.random.seed(0)
x = np.random.rand(300) * 3
x.sort()
y_true = func(x, 1, -4.5, 6, 2)
noise = np.random.rand(300) - 0.5
y = y_true + noise

# Performing gradient descent
a_hat, b_hat, c_hat, d_hat = gradient_descent(x, y)

print("Estimated Parameters (a, b, c, d):", a_hat, b_hat, c_hat, d_hat)
plt.plot(x,y_true,'r-',markersize=2)
plt.plot(x,y,'bo',markersize=2)
ye=func(x, a_hat, b_hat, c_hat, d_hat)
plt.plot(x,ye,'g-',markersize=2)
plt.show()