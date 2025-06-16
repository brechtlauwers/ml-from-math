import numpy as np
from cvxopt import matrix, solvers

from matplotlib import pyplot as plt
import random


class SVM():
    def __init__(self, C=1, kernel='linear'):
        self.C = C
        self.kernel = kernel

    def linear_kernel(self, x_n, x_m):
        return x_n @ x_m

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


# Amount of datapoints and dimensions
N = 20
d = 2

# Keep the data in a certain range
low_limit, up_limit = -5, 5

# Generate different datapoints
x1_n, x2_n = np.random.uniform(low_limit, up_limit, (d, N))
X = np.array(list(zip(x1_n, x2_n)))

# Generate 2 variables for the function f(x) = ax + b
a = np.random.uniform(low_limit, up_limit)
b = np.random.uniform(low_limit, up_limit)

# Add fitting label to the datapoints
y_n = np.sign(x2_n - x1_n*a - b)
y = np.array(y_n)

# Needed to plot the function line
x = np.arange(low_limit, up_limit + 1, 1)
plt.plot(x, a*x+b, c="black")

# Plot positive and negative points
plt.plot(x1_n[y_n > 0], x2_n[y_n > 0], "+", c='r', markersize=10)
plt.plot(x1_n[y_n < 0], x2_n[y_n < 0], "_", c='b', markersize=10)

# Set the limits for the plot
plt.xlim(low_limit, up_limit)
plt.ylim(low_limit, up_limit)


def linear_kernel(x_n, x_m):
    return x_n @ x_m.T


K = linear_kernel(X, X)

P = matrix(np.outer(y, y) * K, tc='d')
q = matrix(np.ones(N) * -1)

G = matrix(np.eye(N) * -1, tc='d')
h = matrix(np.zeros(N))

A = matrix(y.reshape(1, -1), tc='d')
b = matrix(0, tc='d')

solution = solvers.qp(P, q, G, h, A, b)
alphas = np.ravel(solution['x'])

non_zero = alphas > 1e-6
sv = X[non_zero]
sv_y = y[non_zero].reshape(-1, 1)
alphas = alphas[non_zero].reshape(-1, 1)

b = np.mean([
    y_k - ((alphas * sv_y).T @ linear_kernel(sv, x_k.reshape(1, -1))).item()
    for x_k, y_k in zip(sv, sv_y)
])


def predict(X_test):
    K_test = linear_kernel(sv, X_test)
    decision_value = (alphas * sv_y).T @ K_test + b
    return np.sign(decision_value).flatten()

def predict_decision_values(X_test):
    K_test = linear_kernel(sv, X_test)
    decision_value = (alphas * sv_y).T @ K_test + b
    return decision_value.flatten()

# Create a grid to plot the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Evaluate decision function on the grid
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = predict_decision_values(grid_points)
Z = Z.reshape(xx.shape)

# Plot decision boundary and margins
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1],
            alpha=0.5, linestyles=['--', '-', '--'])

plt.scatter(sv[:, 0], sv[:, 1], s=100, facecolors='none', edgecolors='k', linewidths=1.5)

plt.savefig("linear_data.png")
plt.clf()
