import numpy as np
from matplotlib import pyplot as plt
import random
from supervised.svm import SVM


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

# Plot positive and negative points
plt.plot(x1_n[y_n > 0], x2_n[y_n > 0], "+", c='r', markersize=10)
plt.plot(x1_n[y_n < 0], x2_n[y_n < 0], "_", c='b', markersize=10)

# Set the limits for the plot
plt.xlim(low_limit, up_limit)
plt.ylim(low_limit, up_limit)

# Train the SVM on the data
svm = SVM()
svm.fit(X, y)
svm.predict(X)

# Create a grid to plot the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Compute decision function
weights = svm.coef_[0]
Z = (weights[0] * xx + weights[1] * yy + svm.intercept_).reshape(xx.shape)

# Plot decision boundary and margins
plt.contour(xx, yy, Z, colors='gray', levels=[-1, 0, 1],
            alpha=0.8, linestyles=['--', '-', '--'])

# Highlight support vectors
plt.scatter(svm.support_vectors_[:, 0],
            svm.support_vectors_[:, 1], s=100,
            facecolors='none', edgecolors='black', linewidths=1.5)

plt.savefig("svm_linear_decision_boundary.png")
plt.clf()
