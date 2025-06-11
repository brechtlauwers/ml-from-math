import numpy as np


class SVD():
    def transform(self, A):
        n, m = np.shape(A)

        if m >= n:
            B = A.T @ A

            # Step 1: calculate the right singular vectors, matrix V
            eigenvalues, V = np.linalg.eigh(B)

            # Remove negative numbers because of numerical errors
            eigenvalues = np.maximum(eigenvalues, 0)

            # Sort all the eigenvalues and eigenvectors
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            V = V[:, idx]

            # Step 2: calculate the singular value matrix
            s = np.sqrt(eigenvalues)
            # Threshold mask for numerical stability
            non_zero = s > 1e-08
            # S has the same shape as the original matrix A
            S = np.zeros((n, m))
            S[:n, :n] = np.diag(s[non_zero])

            # Step 3: calculate the left singular vectors, matrix U
            U = A @ V[:, non_zero] @ np.diag(1.0 / s[non_zero])

        else:
            B = A @ A.T

            # Step 1: calculate the right singular vectors, matrix V
            eigenvalues, U = np.linalg.eigh(B)

            # Remove negative numbers because of numerical errors
            eigenvalues = np.maximum(eigenvalues, 0)

            # Sort all the eigenvalues and eigenvectors
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            U = U[:, idx]

            # Step 2: calculate the singular value matrix
            s = np.sqrt(eigenvalues)
            # Threshold mask for numerical stability
            non_zero = s > 1e-08
            # S has the same shape as the original matrix A
            S = np.zeros((n, m))
            S[:m, :m] = np.diag(s[non_zero])

            # Step 3: calculate the left singular vectors, matrix U
            V = A.T @ U[:, non_zero] @ np.diag(1.0 / s[non_zero])

        return U, S, V.T
