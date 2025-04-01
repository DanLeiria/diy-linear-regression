# DIY: 📈 Linear Regression from Scratch using Gradient Descent

This project demonstrates a simple, configurable pipeline to estimate the slope and bias of a linear regression model using gradient descent, entirely built from scratch. It includes:
- Data generation (with noise & outliers)
- Gradient descent optimization
- Visualization of the learning process
- Configurable project settings with YAML
- Logging to console and file for easy debugging


## 🧠 Key Features
- Minimal implementation of linear regression with gradient descent
- Parameterized configuration via config.yaml
- Live plotting of cost function optimization
- Modular structure for clarity and reuse
- Logs both to the terminal and app.log

## 🗂️ Project Structure
```
diy-linear-regression
|
├── main.py                   # Main script to run the full pipeline
├── config.yaml               # Configuration file for settings and variables
├── app.log                   # Log output
├── generate_data.py          # Dataset generator with noise/outliers
├── grad_descent.py           # Gradient descent algorithm
├── plotting_data.py          # Plotting utilities
└── README.md
```

## ⚙️ Configuration
You can control your experiment settings using the ``config.yaml`` file (below an example):
```
NUM_DATAPOINTS: 100
SLOPE: 2.0
BIAS: 1.0
NUM_OUTLIERS: 5
LEARNING_RATE: 0.01
EPOCHS: 1000
```

## 🚀 How to Run

1. Clone the repository:
```
git clone https://github.com/DanLeiria/diy-linear-regression.git
cd diy-linear-regression
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Change the config.yaml variables according to what you want to test.

4. Run the main script:
```
python main.py
```

5. Check the output plots and logs (``app.log``) for progress details.

