import numpy as np


class LinearRegr:
    """
    Description
    Parameters
    Output
    """

    # Loss function of the linear regression
    def loss_function(
        self, num_cases: int, y, X, slope_hat: float, intercept_hat: float
    ):
        m = num_cases
        if m <= 0:
            return print("Number of cases cannot be below or equal to 0 (zero).")

        # Estimate the y_hat
        y_hat = slope_hat * X + intercept_hat

        # Calculate loss function
        loss = 1 / m * sum((y - y_hat) ** 2)

        return loss

    # Gradient descent
    def gradient_descent(
        self,
        num_cases,
        X,
        y,
        slope_i: float,
        intercept_i: float,
        learn_rate: float,
    ):
        m = num_cases
        # Calculate the gradients
        slope_grad = -(2 / m) * sum(X * (y - (slope_i * X + intercept_i)))
        intercept_grad = -(2 / m) * sum(y - (slope_i * X + intercept_i))

        # Update slope and intercept
        slope = slope_i - learn_rate * slope_grad
        intercept = intercept_i - learn_rate * intercept_grad

        return slope, intercept

    def estimate_linear(self, X, y, slope_init, intercept_init, learn_rate, epochs):
        # Define number of cases
        if len(X) == len(y):
            m = len(y)
        else:
            print("X and y must have the same size")
            return None

        for ep_i in range(epochs):
            # Calculate the slope and intercept
            slope, intercept = self.gradient_descent(
                num_cases=m,
                X=X,
                y=y,
                slope_i=slope_init,
                intercept_i=intercept_init,
                learn_rate=learn_rate,
            )

            # # Calculate the overall loss
            # loss = LinearRegr.loss_function(
            #     num_cases=m, X=X, y=y, slope_hat=slope, intercept_hat=intercept
            # )

            return slope, intercept
