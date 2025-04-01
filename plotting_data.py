import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from math_functions import loss_func


# Optimization plot
def plot_optimization(cost_list: dict, X, y):
    """
    DESCRIPTION:
          Plot the optimization progress of algorithm.
          Renders a 2D plot between Bias and Slope.
          Shows the track beteen the starting point and end.
    PARAMETERS:
        - cost_list: Dictionary with optimization data (dict)
    OUTPUT:
        - Plot
    """

    # Convert your cost_list to a DataFrame
    df_costs = pd.DataFrame(cost_list)

    # Generate a meshgrid of slope and bias values
    slope_vals = np.linspace(
        df_costs["slope"].min() - 1, df_costs["slope"].max() + 1, 100
    )
    bias_vals = np.linspace(df_costs["bias"].min() - 1, df_costs["bias"].max() + 1, 100)
    S, B = np.meshgrid(slope_vals, bias_vals)

    # Compute cost over the grid
    Z = np.array(
        [[loss_func(slope, bias, X, y) for slope in slope_vals] for bias in bias_vals]
    )

    # Create the contour plot
    plt.figure(figsize=(10, 7))
    contour = plt.contourf(S, B, Z, levels=50, cmap="viridis")
    plt.colorbar(contour, label="Cost")

    # Plot gradient descent path
    plt.plot(
        df_costs["slope"],
        df_costs["bias"],
        color="white",
        marker="o",
        linestyle="-",
        label="Gradient path",
    )
    plt.scatter(
        df_costs["slope"].iloc[0],
        df_costs["bias"].iloc[0],
        color="green",
        label="Start",
        zorder=5,
    )
    plt.scatter(
        df_costs["slope"].iloc[-1],
        df_costs["bias"].iloc[-1],
        color="red",
        label="End",
        zorder=5,
    )

    # Labels and title
    plt.xlabel("Slope")
    plt.ylabel("Bias")
    plt.title("Contour Plot of Cost Function with Gradient Descent Path")
    plt.legend()
    plt.grid(True)
    plt.show()


# Plot regression line from algorithm and the seaborn method
def plot_linear_regressions(X, y):
    """
    DESCRIPTION:
          Plot the optimization progress of algorithm.
          Renders a 2D plot between Bias and Slope.
          Shows the track beteen the starting point and end.
    PARAMETERS:
        - cost_list: Dictionary with optimization data (dict)
    OUTPUT:
        - Plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(x=X, y=y, line_kws={"color": "red", "linewidth": 1.5}, ax=ax)
    plt.show()
