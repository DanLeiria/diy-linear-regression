import logging
import numpy as np

logger = logging.getLogger(__name__)  # Start logger


# Linear equation
def y_func(slope: float, bias: float, X: float):
    """
    DESCRIPTION:
          Calculate y-values based on linear equation.
    PARAMETERS:
        - X: x-values of the cartesian plane (numpy 1D array)
        - slope: defined slope value (float)
        - bias: defined bias value (float)
    OUTPUT:
        - y-values as a numpy 1D array
    """
    return slope * X + bias


# Loss function
def loss_func(slope: float, bias: float, X, y):
    """
    DESCRIPTION:
          Calculate loss function: Mean Squared Error (MSE).
    PARAMETERS:
        - slope: defined slope value (float)
        - bias: defined bias value (float)
        - X: x-values of the cartesian plane (numpy 1D array)
        - y: y-values of the cartesian plane (numpy 1D array)
    OUTPUT:
        - Calculated MSE value
    """
    # Test 1
    if len(X) != len(y):
        logger.error("y and X must have the same length.")

    # Test 2
    if len(X) == 0:
        logger.error("Length of y and X cannot be zero.")

    # Calculate y based on linear regression
    y_calculated = y_func(slope, bias, X)

    # Calculate MSE
    return np.mean((y - y_calculated) ** 2) / 2


# Gradient loss - slope function
def grad_slope_func(slope: float, bias: float, X, y):
    """
    DESCRIPTION:
          Calculate gradient of the loss function: Slope.
    PARAMETERS:
        - slope: defined slope value (float)
        - bias: defined bias value (float)
        - X: x-values of the cartesian plane (numpy 1D array)
        - y: y-values of the cartesian plane (numpy 1D array)
    OUTPUT:
        - Calculated slope gradient
    """
    # Define number of samples
    m = len(X)

    # Test 1
    if m != len(y):
        logger.error("y and X must have the same length.")

    # Test 2
    if m == 0:
        logger.error("Length of y and X cannot be zero.")

    # Calculate y based on linear regression
    y_calculated = y_func(slope, bias, X)

    # Calculate gradient
    return np.dot(X.T, (y_calculated - y)) / m


# Gradient loss - bias function
def grad_bias_func(slope: float, bias: float, X, y):
    """
    DESCRIPTION:
          Calculate gradient of the loss function: Bias.
    PARAMETERS:
        - slope: defined slope value (float)
        - bias: defined bias value (float)
        - X: x-values of the cartesian plane (numpy 1D array)
        - y: y-values of the cartesian plane (numpy 1D array)
    OUTPUT:
        - Calculated bias gradient
    """
    # Define number of samples
    m = len(X)

    # Test 1
    if m != len(y):
        logger.error("y and X must have the same length.")

    # Test 2
    if m == 0:
        logger.error("Length of y and X cannot be zero.")

    # Calculate y based on linear regression
    y_calculated = y_func(slope, bias, X)

    # Calculate gradient
    return np.sum(y_calculated - y) / m


if __name__ == "__main__":
    import numpy as np
    from generate_data import create_dataset

    NUM_DATAPOINTS = 100
    SLOPE = 1.0
    BIAS = 0.0
    NUM_OUTLIERS = 10

    X, y = create_dataset(
        num_cases=NUM_DATAPOINTS,
        slope=SLOPE,
        bias=BIAS,
        num_outliers=NUM_OUTLIERS,
    )

    print(y_func(slope=SLOPE, bias=BIAS, X=X))
    print(grad_slope_func(slope=SLOPE, bias=BIAS, X=X, y=y))
    print(grad_bias_func(slope=SLOPE, bias=BIAS, X=X, y=y))
