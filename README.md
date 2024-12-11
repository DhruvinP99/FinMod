# Stock Price Prediction with LSTM

This project builds a machine learning model to predict stock prices using historical data from Yahoo Finance and an LSTM (Long Short-Term Memory) model. The goal is to predict future stock prices with a low root mean squared error (RMSE) while training the model for 100 epochs, even if early stopping conditions are met.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Streamlit App](#streamlit-app)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project is part of my Big Data Analysis and Deep Learning classes at Loyola University Chicago. The main objective is to predict future stock prices using historical stock data with an LSTM model. The goal is to achieve an RMSE close to 1 or 5 and train the model over 100 epochs without early stopping, regardless of convergence in loss values.

**Key Objectives:**
- Use an LSTM-based model to predict stock prices.
- Achieve a target RMSE near 1 or 5.
- Train the model over 100 epochs without early stopping.
- Output predictions as graphs and CSV files for further analysis.

## Dataset

This project uses historical stock price data obtained from Yahoo Finance via the `yfinance` library. The dataset includes:
- Daily closing prices
- Timestamps for each trading day

Additional preprocessing steps include:
- **Normalization**: The stock prices are normalized using the `StandardScaler` from `sklearn`.
- **Sliding Window Approach**: Data is transformed into sequences of past stock prices for the LSTM model.

## Model Architecture

The LSTM model architecture used in this project consists of the following layers:
1. **LSTM Layers**: The model contains multiple LSTM layers to capture temporal dependencies in stock price data.
2. **Dense Layer**: A fully connected layer outputs the final prediction for the stock price.

The model is trained using the **RMSE (Root Mean Squared Error)** as the primary evaluation metric, and we aim for the lowest possible RMSE through hyperparameter tuning and optimization.

### Model Details:
- **Input Layer**: Accepts normalized stock prices.
- **LSTM Layers**: Two LSTM layers with 32 hidden units.
- **Dropout**: 0.4 dropout rate to avoid overfitting.
- **Optimizer**: Adam optimizer with a learning rate of `1e-5`.
- **Loss Function**: Huber Loss for robust learning.
- **Epochs**: 100 epochs with a learning rate scheduler.

### Key Features:
- **Training without Early Stopping**: The model is trained over 100 epochs, even if loss convergence is observed.
- **Grid Search for Hyperparameters**: Fine-tuning of hyperparameters is done via grid search and cross-validation.

## Installation

### Requirements:
- Python 3.x
- Install required libraries via pip:

```bash
pip install -r requirements.txt
