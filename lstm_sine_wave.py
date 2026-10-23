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
    noisy_sin_series = torch.sin(t * 0.05) * torch.randn(n)
    # plot_data(t, noisy_sin_series)
    return noisy_sin_series

def normalize_data(series: torch.Tensor) -> torch.Tensor:
    """Normalize entire data based on training sample (80%)

    Args:
        series (torch.Tensor): Entire data

    Returns:
        torch.Tensor: Normalized data based on training sample
    """
    split = int(0.8 * len(series))
    mean, std = series[:split].mean(), series[:split].std()
    series = (series - mean) / std
    return series


class WindowDataset(Dataset):
    def __init__(self, series, seq_len):
        self.series = series
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.series) - self.seq_len
    
    def __getitem__(self, i):
        # X = last i number of values
        # y = next item after the sequence length starting from i
        x = self.series[i : i + self.seq_len].unsqueeze(dim=-1)        # shape of (seq_len, 1) -> seq_len samples and 1 feature
        print(f"Shape of x: {x.shape}")
        y = self.series[i + self.seq_len].unsqueeze(dim=-1)            # shape of (1, )
        print(f"Shape of y: {y.shape}")
        return x, y

if __name__ == "__main__":
    print(f"Torch version: {torch.__version__}")
    print(f"Device: {DEVICE}")
    
    # Create noisy sine wave data
    series = create_data()
    # Normalize data
    series = normalize_data(series)
    
    split = int(0.8 * len(series))
    
    # Train is from 0 to 3999
    train_ds = WindowDataset(series[:split], SEQ_LEN)
    
    # Test is from 3950 to 4999 (because the first predicted value is at indexd 4000, so to predict it we need values from 3950)
    test_ds = WindowDataset(series[split - SEQ_LEN:], SEQ_LEN)
    
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=64)