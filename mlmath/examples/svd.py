from mlfoundations.svd import SVD
import numpy as np


def main():
    # Because of numerical instabilities, very small values could occur.
    # In most cases, it should be 0
    
    # Example 1
    A = np.array([[1, 0, 1],
                  [-2, 1, 0]])
    U, S, V_T = SVD().transform(A)
    print("Original:")
    print(A)
    print("Reconstructed with SVD:")
    print(U @ S @ V_T)

    # Example 2
    B = np.array([
        [5, 4, 1],
        [5, 5, 0],
        [0, 0, 5],
        [1, 0, 4]
    ])
    U, S, V_T = SVD().transform(B)
    print("Original:")
    print(B)
    print("Reconstructed with SVD:")
    print(U @ S @ V_T)


if __name__ == "__main__":
    main()
