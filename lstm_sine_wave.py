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
    # Create array of length N_POINTS from 0 to N_POINTS - 1, of step=1 -> set the tensor values to float32
    t = torch.arange(n, dtype=torch.float32)
    # Create random noisy sin wave
    noisy_sin = torch.sin(t * 0.05) * torch.randn(n)
    plot_data(t, noisy_sin)

if __name__ == "__main__":
    print(f"Torch version: {torch.__version__}")
    print(f"Device: {DEVICE}")
    create_data()
    plot_data()