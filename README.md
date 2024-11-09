# Stock Price Prediction with LSTM

This project aims to build a machine learning model to predict stock prices using historical data from Yahoo Finance and an LSTM (Long Short-Term Memory) model. The main goal is to achieve a low root mean squared error (RMSE) while training the model for a full 100 epochs, even if early stopping conditions are met.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The project is part of my Big Data Analysis and Deep Learning classes at Loyola University Chicago. The primary objective is to predict future stock prices using historical stock price data. This README provides a complete overview of the setup, model training process, and key evaluation metrics.

**Key Objectives:**
- Use an LSTM-based model to predict stock prices.
- Achieve a target RMSE near 1 or 5.
- Train the model over 100 epochs without early stopping, regardless of convergence in loss values.
- Output prediction results as graphs and CSV files.

## Dataset

This project utilizes historical stock price data obtained from Yahoo Finance via the `yfinance` library. The data includes:
- Daily closing prices
- Timestamps for each trading day

Additional pre-processing steps ensure the data is normalized and structured for input into the LSTM model.

## Model Architecture

The LSTM model architecture consists of multiple layers to capture temporal dependencies in stock price data effectively:
1. **LSTM Layers**: To learn from the sequence data over time.
2. **Dense Layers**: To output the final predictions.

We use RMSE as the main evaluation metric and aim for the lowest possible error by fine-tuning hyperparameters through custom n-fold cross-validation and a manual grid search.

## Installation

To run this project, you'll need Python and several libraries. Follow the steps below to set up your environment:



