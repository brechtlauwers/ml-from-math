import numpy as np
from cvxopt import matrix, solvers

class SVM():
    def __init__(self, C=0, kernel='linear'):
        self.C = C
        self.kernel = kernel

        self.coef_ = []
        self.intercept_ = 0
        self.support_vectors_ = []

    def linear_kernel(self, x_n, x_m):
        return x_n @ x_m.T

    def fit(self, X, y):
        # Full kernel matrix
        K = self.linear_kernel(X, X)

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
            b_s = y_s - y_alphas @ self.linear_kernel(self.sv, x_s.reshape(1, -1))
            total_b.append(np.ravel(b_s)[0])

        self.b = np.mean(total_b)

        # Set variables for accessibility
        self.coef_ = (self.alphas * self.sv_y).T @ self.sv
        self.intercept_ = self.b
        self.support_vectors_ = self.sv

    def predict(self, X_test):
        K_test = self.linear_kernel(self.sv, X_test)
        decision_value = (self.alphas * self.sv_y).T @ K_test + self.b
        return np.sign(decision_value).flatten()

    def predict_decision_values(self, X_test):
        K_test = self.linear_kernel(self.sv, X_test)
        decision_value = (self.alphas * self.sv_y).T @ K_test + self.b
        return decision_value.flatten()



