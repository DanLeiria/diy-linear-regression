from generate_data import create_dataset
from grad_descent import gradient_descent
from plotting_data import plot_optimization

import numpy as np
import logging  # Logs of the scripts
import yaml  # Load project settings from a .yaml configuration file


### ================================================================== ###
###                         LOGGING CONFIG                             ###
### ================================================================== ###
"""
Prepare configuration of the logs.
"""

logging.basicConfig(
    level=logging.INFO,  # Could be DEBUG, WARNING, ERROR, CRITICAL
    format="%(asctime)s: [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log", mode="w"),  # Logs to a file
        logging.StreamHandler(),  # Logs to console
    ],
)

# Create a logger
logger = logging.getLogger(__name__)

logger.info("Starting main...")

### ================================================================== ###
###                          GENERATE DATA                             ###
### ================================================================== ###
""" 
Generate data based on the slope, bias, number of datapoints and outliers.
"""

# Load config file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Test
if not config:
    logger.error("Couldn't load config file.")

X, y = create_dataset(
    num_cases=config["NUM_DATAPOINTS"],
    slope=config["SLOPE"],
    bias=config["BIAS"],
    num_outliers=config["NUM_OUTLIERS"],
)


### ================================================================== ###
###                 RUN DEVELOPED LINEAR REGRESSION                    ###
### ================================================================== ###
"""
Run the developed algorithm fit a linear regression using gradient descent.
"""

# Random initialization of the slope and the bias values
slope_init = np.random.rand(1)
bias_init = np.random.rand(1)

# Variables for the algorithm gradient descent
LEARNING_RATE = config["LEARNING_RATE"]
EPOCHS = config["EPOCHS"]


slope_final, bias_final, cost_list = gradient_descent(
    X=X,
    y=y,
    slope_init=slope_init,
    bias_init=bias_init,
    learn_rate=LEARNING_RATE,
    epochs=EPOCHS,
)

logger.info("Estimation concluded.")


### ================================================================== ###
###                         OPTIMIZATION PLOT                          ###
### ================================================================== ###

plot_optimization(cost_list=cost_list, X=X, y=y)


logger.info("Finished main.")
