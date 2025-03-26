from generate_data import create_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

### ================================================================== ###
###                          GENERATE DATA                             ###
### ================================================================== ###

NUM_DATAPOINTS = 100
SLOPE = 1.0
INTERCEPT = 0.0
NUM_OUTLIERS = 10

X, y = create_dataset(
    num_cases=NUM_DATAPOINTS,
    slope=SLOPE,
    intercept=INTERCEPT,
    num_outliers=NUM_OUTLIERS,
)


### ================================================================== ###
###                 RUN DEVELOPED LINEAR REGRESSION                    ###
### ================================================================== ###

slope_init = np.random.rand(1)
intercept_init = np.random.rand(1)

LEARNING_RATE = 0.001
EPOCHS = 100


### ================================================================== ###
###                      RUN PACKAGE ESTIMATION                        ###
### ================================================================== ###

fig, ax = plt.subplots(figsize=(8, 6))
sns.regplot(x=X, y=y, line_kws={"color": "red", "linewidth": 1.5}, ax=ax)
plt.show()
