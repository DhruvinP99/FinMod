import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

# Get the stock ticker input from the user
ticker = input("Enter the stock ticker (e.g., AAPL, MSFT, AMZN): ").strip()

# Configuration dictionary
config = {
    "data": {
        "window_size": 20,
        "train_split_size": 0.80,
    },
    "model": {
        "input_size": 1,
        "num_lstm_layers": 2,
        "lstm_size": 32,
        "dropout": 0.2,
    },
    "training": {
        "device": "cpu",
        "batch_size": 64,
        "num_epoch": 100,
        "learning_rate": 0.01,
        "scheduler_step_size": 40,
    }
}

# Fetch stock data dynamically for the last 1 year
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

print(f"Fetching data for {ticker} from {start_date} to {end_date}...")

# Download the stock data using yfinance
try:
    data = yf.download(ticker, start=start_date, end=end_date, timeout=30)
    if data.empty:
        print(f"Failed to fetch data for {ticker}. Please check the ticker.")
        exit()
    else:
        print("Data fetched successfully!")
        print(data.head())
except Exception as e:
    print(f"Error fetching data: {e}")
    exit()

# Normalize and preprocess data
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

scaler = Normalizer()
data_close_price = data['Adj Close'].values
normalized_data_close_price = scaler.fit_transform(data_close_price)

def prepare_data_x(x, window_size):
    n_row = x.shape[0] - window_size + 1
    output = np.lib.stride_tricks.as_strided(x, shape=(n_row, window_size), strides=(x.strides[0], x.strides[0]))
    return output[:-1], output[-1]

def prepare_data_y(x, window_size):
    return x[window_size:]

data_x, data_x_unseen = prepare_data_x(normalized_data_close_price, config["data"]["window_size"])
data_y = prepare_data_y(normalized_data_close_price, config["data"]["window_size"])

split_index = int(data_y.shape[0] * config["data"]["train_split_size"])
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

train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=True)
val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=False)

# LSTM Model
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

model = LSTMModel(
    input_size=config["model"]["input_size"],
    hidden_layer_size=config["model"]["lstm_size"],
    num_layers=config["model"]["num_lstm_layers"],
    output_size=1,
    dropout=config["model"]["dropout"]
).to(config["training"]["device"])

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.1)

# Training loop
def train_epoch(dataloader, model, criterion, optimizer, device, is_training):
    total_loss = 0
    model.train() if is_training else model.eval()
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(is_training):
            outputs = model(x)
            loss = criterion(outputs.squeeze(), y)
            if is_training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

for epoch in range(config["training"]["num_epoch"]):
    train_loss = train_epoch(train_dataloader, model, criterion, optimizer, config["training"]["device"], True)
    val_loss = train_epoch(val_dataloader, model, criterion, optimizer, config["training"]["device"], False)
    scheduler.step()
    print(f"Epoch {epoch+1}/{config['training']['num_epoch']} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

# Predictions
model.eval()
pred_train = []
for x, _ in train_dataloader:
    x = x.to(config["training"]["device"])
    pred_train.extend(model(x).cpu().detach().numpy())

pred_val = []
for x, _ in val_dataloader:
    x = x.to(config["training"]["device"])
    pred_val.extend(model(x).cpu().detach().numpy())

# Convert predictions to NumPy arrays
pred_train = np.array(pred_train).flatten()
pred_val = np.array(pred_val).flatten()

# Evaluate performance using metrics
train_mse = mean_squared_error(data_y_train, pred_train)
val_mse = mean_squared_error(data_y_val, pred_val)

train_rmse = np.sqrt(train_mse)
val_rmse = np.sqrt(val_mse)

train_mae = mean_absolute_error(data_y_train, pred_train)
val_mae = mean_absolute_error(data_y_val, pred_val)

train_r2 = r2_score(data_y_train, pred_train)
val_r2 = r2_score(data_y_val, pred_val)

train_ev = explained_variance_score(data_y_train, pred_train)
val_ev = explained_variance_score(data_y_val, pred_val)

# Print evaluation metrics
print(f"Train MSE: {train_mse:.6f}")
print(f"Validation MSE: {val_mse:.6f}")

print(f"Train RMSE: {train_rmse:.6f}")
print(f"Validation RMSE: {val_rmse:.6f}")

print(f"Train MAE: {train_mae:.6f}")
print(f"Validation MAE: {val_mae:.6f}")

print(f"Train R²: {train_r2:.6f}")
print(f"Validation R²: {val_r2:.6f}")

print(f"Train Explained Variance: {train_ev:.6f}")
print(f"Validation Explained Variance: {val_ev:.6f}")

# Predict next day's price
last_window = torch.tensor(data_x_unseen).float().to(config["training"]["device"]).unsqueeze(0).unsqueeze(2)
next_day_price = scaler.inverse_transform(model(last_window).cpu().detach().numpy())

print(f"Predicted price for next day: {next_day_price[0][0]}")
