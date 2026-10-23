"""
Train a LSTM using PyTorch to predict next value of a noisy sine wave

Self-contained (no dataset needed). Run with:
    pip install torch
    python lstm_sine_wave.py
"""
import math
import torch
import torch.nn as nn


# Set seed
torch.manual_seed(42)

# Set device agnostic 
# Use GPU is available, if not use Apple Silicon, if not, use CPU
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

if __name__ == "__main__":
    print(f"Torch version: {torch.__version__}")
    print(f"Device: {DEVICE}")

