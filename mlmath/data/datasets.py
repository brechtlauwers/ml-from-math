import numpy as np

def make_linear(n_samples=40, n_features=2, range_box=(-5, 5), noise_ratio=0):
    low_limit, up_limit = range_box

    # Generate random datapoints with the specified dimension
    X = np.random.uniform(low_limit, up_limit, (n_samples, n_features))
    # Generate random coefficients for the separator function
    coefficients = np.random.uniform(low_limit, up_limit, n_features + 1)

    # Classification function
    weighted_sum = X @ coefficients[:-1] + coefficients[-1]
    y = np.sign(weighted_sum)

    # Add a certain amount of noise by randomly switching some labels
    if noise_ratio != 0:
        n_noisy = int(noise_ratio * len(y))
        flip_indices = np.random.choice(len(y), n_noisy, replace=False)
        y[flip_indices] *= -1  # Flip labels

    return X, y

