import numpy as np
import random


def create_dataset(num_cases: int, slope: float, intercept: float, num_outliers: int):
    # Generate X-values (random)
    X = np.random.rand(num_cases)

    # Estimate y-values
    y = slope * X + intercept

    # Add outliers
    if num_outliers > 0:
        # Randomly pick unique indices
        outlier_indices = np.random.choice(num_cases, size=num_outliers, replace=False)

        # Add large noise to those indices
        noise = np.random.normal(
            loc=0, scale=1, size=num_outliers
        )  # You can increase scale for more extreme outliers
        y[outlier_indices] += noise

    return X, y


if __name__ == "__main__":
    X, y = create_dataset(num_cases=100, slope=1.0, intercept=0.0, num_outliers=10)
    print(y)
