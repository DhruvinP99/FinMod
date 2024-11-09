Overview
This project utilizes a Long Short-Term Memory (LSTM) neural network to predict stock prices based on historical stock price data collected via Yahoo Finance (yfinance). The model aims to provide accurate predictions by training on daily price data and evaluating the Root Mean Squared Error (RMSE) to assess model performance.

Table of Contents
-Project Structure
-Data Collection
-Model Architecture
-Training Process
-Evaluation
-Results
-Future Work
-Dependencies
-How to Run

Project Structure
data/: Folder for storing raw and processed stock data.
README.md: Project overview and instructions.

Data Collection
Historical stock price data is retrieved from Yahoo Finance using the yfinance library. Key features include:

Date
Open, High, Low, Close prices
Volume
The data is split into training and test sets for model evaluation.

Model Architecture
The model uses an LSTM neural network, well-suited for time series prediction due to its ability to capture long-term dependencies in data. Key layers in the model:

LSTM layer(s): Capture sequential patterns in stock prices.
Dense layer: Provides final predictions.
Training Process
The model trains for a specified number of epochs, with loss calculated at each epoch. Early stopping criteria are disabled to ensure the model trains fully for 100 epochs. Key parameters include:

Epochs: 100
Loss Function: Mean Squared Error
Optimizer: Adam
Evaluation
The model is evaluated using RMSE, aiming for a target RMSE near 1 or 5, as per the project goals. Validation losses are also tracked but do not impact training duration.

Results
The model outputs include:

Predicted vs. actual stock prices over time
RMSE scores
CSV files with predictions for further analysis

Future Work
Integrate advanced hyperparameter tuning techniques.
Experiment with additional data features or alternative architectures.
Enhance prediction visualization using a heatmap.

Dependencies
Python 3.7+
tensorflow
yfinance
pandas
matplotlib
