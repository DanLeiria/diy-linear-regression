from math_functions import grad_slope_func, grad_bias_func, loss_func
import logging

logger = logging.getLogger(__name__)  # Start logger


# Gradient descent (GD)
def gradient_descent(
    X, y, slope_init: float, bias_init: float, learn_rate: float, epochs: int
):
    """
    DESCRIPTION:
        - Apply gradient descent (GD) algorithm to estimate the
          slope and bias.
    PARAMETERS:
        - X: x-values of the cartesian plane (numpy 1D array)
        - y: y-values of the cartesian plane (numpy 1D array)
        - slope_init: starting slope value (float)
        - bias_init: starting bias value (float)
        - learn_rate: learning rate of the GD (float)
        - epochs: Number of the epochs of the GD (int)
    OUTPUT:
        - slope: Final slope
        - bias: Final slope
    """
    # Define number of samples
    m = len(X)

    # Test 1
    if m != len(y):
        logger.error("y and X must have the same length.")

    # Test 2
    if m == 0:
        logger.error("Length of y and X cannot be zero.")

    # Assign starting values of slope, bias, and cost
    slope_i = slope_init
    bias_i = bias_init
    cost_list = []

    for i in range(epochs):
        # Update slope and intercept
        slope_i = slope_i - learn_rate * grad_slope_func(
            slope=slope_i, bias=bias_i, X=X, y=y
        )
        bias_i = bias_i - learn_rate * grad_bias_func(
            slope=slope_i, bias=bias_i, X=X, y=y
        )

        # Check loss progression
        if i % 100 == 0:
            cost_i = loss_func(slope=slope_i, bias=bias_i, X=X, y=y)

            # Append a dictionary with values
            cost_list.append(
                {"iteration": i, "slope": slope_i, "bias": bias_i, "cost": cost_i}
            )

    return slope_i, bias_i, cost_list


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

    slope_init = np.random.rand(1)
    bias_init = np.random.rand(1)

    LEARNING_RATE = 0.1
    EPOCHS = 100000

    slope, bias, cost_list = gradient_descent(
        X=X,
        y=y,
        slope_init=slope_init,
        bias_init=bias_init,
        learn_rate=LEARNING_RATE,
        epochs=EPOCHS,
    )

    print(slope)
    print(bias)
    print(cost_list)
