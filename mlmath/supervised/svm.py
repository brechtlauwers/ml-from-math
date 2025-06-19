import numpy as np
from cvxopt import matrix, solvers
from mlfoundations.kernels import linear_kernel, polynomial_kernel


class SVM():
    def __init__(self, C=0, kernel='linear', degree=3, gamma=None):
        self.C = C

        if kernel == 'linear':
            self.kernel = linear_kernel
        elif kernel == 'poly':
            self.kernel = lambda x_n, x_m: polynomial_kernel(x_n, x_m, degree, gamma)

        # Extra parameters
        self.coef_ = None
        self.intercept_ = None
        self.support_vectors_ = None

    def fit(self, X, y):
        # Full kernel matrix
        K = linear_kernel(X, X)

        N = X.shape[0]

        # Construct needed matrices for quadratic programming
        P = matrix(np.outer(y, y) * K, tc='d')
        q = matrix(np.ones(N) * -1)

        G = matrix(np.eye(N) * -1, tc='d')
        h = matrix(np.zeros(N))

        A = matrix(y.reshape(1, -1), tc='d')
        b = matrix(0, tc='d')

        # Solve QP problem
        solution = solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(solution['x'])

        # Set tolerance to find support vectors with alpha = 0
        non_zero = alphas > 1e-6
        self.sv = X[non_zero]
        self.sv_y = y[non_zero].reshape(-1, 1)
        self.alphas = alphas[non_zero].reshape(-1, 1)

        # Use mean of all SVs to calculate b more stably
        total_b = []
        for x_s, y_s in zip(self.sv, self.sv_y):
            y_alphas = (self.alphas * self.sv_y).T
            b_s = y_s - y_alphas @ linear_kernel(self.sv, x_s.reshape(1, -1))
            total_b.append(np.ravel(b_s)[0])

        self.b = np.mean(total_b)

        # Set variables for accessibility
        self.coef_ = (self.alphas * self.sv_y).T @ self.sv
        self.intercept_ = self.b
        self.support_vectors_ = self.sv

    def predict(self, X_test):
        K_test = linear_kernel(self.sv, X_test)
        decision_value = (self.alphas * self.sv_y).T @ K_test + self.b
        return np.sign(decision_value).flatten()

    def predict_decision_values(self, X_test):
        K_test = linear_kernel(self.sv, X_test)
        decision_value = (self.alphas * self.sv_y).T @ K_test + self.b
        return decision_value.flatten()
