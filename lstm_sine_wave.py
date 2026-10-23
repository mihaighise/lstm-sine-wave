"""
Train a LSTM using PyTorch to predict next value of a noisy sine wave

Self-contained (no dataset needed). Run with:
    pip install torch
    python lstm_sine_wave.py
"""
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# Set seed
torch.manual_seed(42)

# Set device agnostic 
# Use GPU is available, if not use Apple Silicon, if not, use CPU
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# ----------------------------------------------------------------------------
# 1. Data: turn one long series into (window of past values -> next value)
# ----------------------------------------------------------------------------
SEQ_LEN = 50        # how many past steps the LSTM sees
N_POINTS = 5000

def plot_data(t: torch.Tensor, y: torch.Tensor):
    plt.figure(figsize=(10, 7))
    plt.plot(t[:500].numpy(), y[:500].numpy())
    plt.xlabel("Values")
    plt.ylabel("Noisy sin wave")
    plt.title("Sine Wave (500 Points)")
    plt.grid(True)
    plt.show()

def create_data(n=N_POINTS):
    """Create noisy sine wave dataset

    Args:
        n (_type_, optional): Number of samples to be created. Defaults to N_POINTS.

    Returns:
        _type_: Torch array of noisy sine wave samples
    """
    # Create array of length N_POINTS from 0 to N_POINTS - 1, of step=1 -> set the tensor values to float32
    t = torch.arange(n, dtype=torch.float32)
    # Create random noisy sin wave
    noisy_sin_series = torch.sin(t * 0.05) + 0.1 * torch.randn(n)
    # plot_data(t, noisy_sin_series)
    return noisy_sin_series


class WindowDataset(Dataset):
    def __init__(self, series, seq_len):
        self.series = series
        self.seq_len = seq_len
    
    # The first SEQ_LEN values cannot be targets
    def __len__(self):
        return len(self.series) - self.seq_len
    
    def __getitem__(self, i):
        # X = last i number of values
        # y = next item after the sequence length starting from i
        x = self.series[i : i + self.seq_len].unsqueeze(dim=-1)        # shape of (seq_len, 1) -> seq_len samples and 1 feature
        # print(f"Shape of x: {x.shape}")
        y = self.series[i + self.seq_len].unsqueeze(dim=-1)            # shape of (1, )
        # print(f"Shape of y: {y.shape}")
        return x, y
    
# Create noisy sine wave data
series = create_data()

# Normalize data based on training data
split = int(0.8 * len(series))
mean, std = series[:split].mean(), series[:split].std()
series = (series - mean) / std
    
split = int(0.8 * len(series))
    
# Train is from 0 to 3999
train_ds = WindowDataset(series[:split], SEQ_LEN)
    
# Test is from 3950 to 4999 (because the first predicted value is at indexd 4000, so to predict it we need values from 3950)
test_ds = WindowDataset(series[split - SEQ_LEN:], SEQ_LEN)
    
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=64)
    

# ------------------------------------------------------------------------------------------
# 2. Model & Training: Train a LSTM to predict the next value given the past 50 values
# ------------------------------------------------------------------------------------------
class LSTMForecaster(nn.Module):
    def __init__(self, n_features=1, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,                          # size of x(t)
            hidden_size=hidden_size,                        # size of h(t) and c(t) 
            num_layers=num_layers,                          # stacked LSTMs
            batch_first=True,                               # tensors are (batch, time, features)
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(
            in_features=hidden_size,
            out_features=1
        )
    
    def forward(self, x):
        # out: h(t) for every step  -> (batch, seq_len, hidden)
        # (h_n, c_n): final hidden and cell state per layer
        out, (h_n, c_n) = self.lstm(x)
        last = out[:, -1, :]            # h at the final step
        return self.head(last)          # (batch, 1)
    

# Initialize model
model = LSTMForecaster().to(DEVICE)
print(model)
print("trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# ------------------------------------------------------------------------------------------
# 3. Training loop
# ------------------------------------------------------------------------------------------
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

EPOCHS = 100

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    # train_dl is the array of batches
    for X, y in train_dl:
        # Put data on target device
        X, y = X.to(DEVICE), y.to(DEVICE)
        
        # Forward pass
        y_pred = model(X)
        
        # Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        
        # Optimizer zero grad
        optimizer.zero_grad()
        
        # Backpropagation
        loss.backward()
        
        # Gradient descent
        optimizer.step()
        
    train_loss /= len(train_dl)
    
    # Testing loop
    model.eval()
    test_loss = 0.0
    
    with torch.no_grad():
        for X, y in test_dl:
            X, y = X.to(DEVICE), y.to(DEVICE)
            y_pred = model(X)
            
            loss = loss_fn(y_pred, y)
            test_loss += loss
            
    test_loss /= len(test_dl)  

    print(f"epoch {epoch}  train MSE {train_loss:.4f}  val MSE {test_loss:.4f}")

torch.save(model.state_dict(), "lstm_model.pt")
            
# ---------------------------------------------------------------------------
# 4. Use the trained model
# ---------------------------------------------------------------------------
model.load_state_dict(torch.load("lstm_model.pt", map_location=DEVICE))
model.eval()
with torch.no_grad():
    window = series[-SEQ_LEN:].view(1, SEQ_LEN, 1).to(DEVICE)
    pred = model(window).item() * std + mean      # undo normalisation
print(f"next-value prediction: {pred:.3f}")

if __name__ == "__main__":
    print(f"Torch version: {torch.__version__}")
    print(f"Device: {DEVICE}")