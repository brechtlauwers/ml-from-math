import numpy as np


def linear_kernel(x_n, x_m):
    """K(x, y) = x^T y"""
    return x_n @ x_m.T

def polynomial_kernel(x_n, x_m, degree=3, gamma=None):

    pass
    
