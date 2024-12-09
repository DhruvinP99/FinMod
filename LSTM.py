import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yfinance as yf
from datetime import datetime, timedelta

# Importing your pre-defined model and normalization class
class Normalizer:
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=0)
        self.sd = np.std(x, axis=0)
        return (x - self.mu) / self.sd

    def inverse_transform(self, x):
        return (x * self.sd) + self.mu


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(self.dropout(lstm_out[:, -1, :]))
        return output


# Streamlit UI
st.title("Stock Price Prediction using LSTM")
st.markdown("This app uses an LSTM model to predict the next day's stock price based on historical data.")

# Sidebar Inputs
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, AMZN)", value="AAPL")
window_size = st.sidebar.slider("Window Size for Prediction", min_value=10, max_value=50, value=20, step=5)
train_split = st.sidebar.slider("Training Split Ratio", min_value=0.6, max_value=0.9, value=0.8, step=0.05)
num_epochs = st.sidebar.slider("Number of Training Epochs", min_value=10, max_value=500, value=100, step=10)
learning_rate = st.sidebar.slider("Learning Rate", min_value=0.001, max_value=0.1, value=0.01, step=0.001)

# Fetch stock data
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

st.write(f"Fetching data for {ticker} from {start_date} to {end_date}...")
try:
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data.empty:
        st.error("Failed to fetch data. Please check the ticker.")
    else:
        st.success("Data fetched successfully!")
        st.line_chart(data["Adj Close"])
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# Normalize and preprocess data
data_close_price = data['Adj Close'].values
scaler = Normalizer()
normalized_data_close_price = scaler.fit_transform(data_close_price)


def prepare_data_x(x, window_size):
    n_row = x.shape[0] - window_size + 1
    output = np.lib.stride_tricks.as_strided(x, shape=(n_row, window_size), strides=(x.strides[0], x.strides[0]))
    return output[:-1], output[-1]


def prepare_data_y(x, window_size):
    return x[window_size:]


data_x, data_x_unseen = prepare_data_x(normalized_data_close_price, window_size)
data_y = prepare_data_y(normalized_data_close_price, window_size)

split_index = int(data_y.shape[0] * train_split)
data_x_train, data_x_val = data_x[:split_index], data_x[split_index:]
data_y_train, data_y_val = data_y[:split_index], data_y[split_index:]

# Dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.x = np.expand_dims(x, axis=2).astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

train_dataloader = DataLoader(dataset_train, batch_size=64, shuffle=True)
val_dataloader = DataLoader(dataset_val, batch_size=64, shuffle=False)

# Model setup
device = "cpu"
model = LSTMModel(input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
if st.button("Train Model"):
    progress_bar = st.progress(0)
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs.squeeze(), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_dataloader))

        # Validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_dataloader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs.squeeze(), y)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_dataloader))

        progress_bar.progress((epoch + 1) / num_epochs)

    st.write("Training Complete!")
    st.line_chart({"Train Loss": train_losses, "Validation Loss": val_losses})

    # Predict next day's price
    last_window = torch.tensor(data_x_unseen).float().to(device).unsqueeze(0).unsqueeze(2)
    next_day_price = scaler.inverse_transform(model(last_window).cpu().detach().numpy())

    current_price = data_close_price[-1]
    st.write(f"**Current Price**: ${current_price:.2f}")
    st.write(f"**Predicted Price for Next Day**: ${next_day_price[0][0]:.2f}")

    # Compute evaluation metrics
    mse = np.mean((next_day_price[0][0] - current_price) ** 2)
    rmse = np.sqrt(mse)
    mae = np.abs(next_day_price[0][0] - current_price)
    r2 = 1 - (mse / np.var([current_price]))

    st.write("### Evaluation Metrics")
    st.write(f"**MSE**: {mse:.6f}")
    st.write(f"**RMSE**: {rmse:.6f}")
    st.write(f"**MAE**: {mae:.6f}")
    st.write(f"**R²**: {r2:.6f}")
