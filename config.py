"""
Configuration file for SNN-ECG anomaly detection experiments.
"""

import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data configuration
DATA_DIR = 'data/mit-bih'
SAMPLE_RATE = 360  # MIT-BIH sampling rate
SEGMENT_LENGTH = 256  # Samples per segment
N_CLASSES = 2  # Normal vs Anomaly

# SNN Architecture
SNN_CONFIG = {
    'input_size': 256,
    'hidden_sizes': [128, 64, 32],
    'output_size': 2,
    'time_steps': 100,
    'tau_mem': 10.0,  # Membrane time constant
    'tau_syn': 5.0,   # Synaptic time constant
    'threshold': 1.0,
    'dt': 1.0,
}

# CNN Baseline Architecture
CNN_CONFIG = {
    'input_channels': 1,
    'hidden_channels': [32, 64, 128],
    'kernel_size': 5,
    'output_size': 2,
}

# STDP Configuration
STDP_CONFIG = {
    'a_plus': 0.01,   # LTP amplitude
    'a_minus': 0.01,  # LTD amplitude
    'tau_plus': 20.0, # LTP time constant
    'tau_minus': 20.0, # LTD time constant
}

# Training configuration
TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 50,
    'k_folds': 5,
    'early_stopping_patience': 10,
}

# Evaluation thresholds
PERFORMANCE_TARGETS = {
    'sensitivity': 0.80,
    'specificity': 0.90,
    'macro_f1': 0.75,
}

# Noise robustness test
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.15, 0.2]  # Noise standard deviation levels for testing
