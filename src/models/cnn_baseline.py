"""
CNN baseline model for fair comparison with SNN.
"""

import torch
import torch.nn as nn


class CNNBaseline(nn.Module):
    """
    1D CNN baseline for ECG anomaly detection.
    Designed for fair comparison with SNN architecture.
    
    Args:
        input_channels: Number of input channels (typically 1 for ECG)
        hidden_channels: List of hidden channel sizes
        kernel_size: Convolution kernel size
        output_size: Number of output classes
    """
    
    def __init__(self, input_channels=1, hidden_channels=[32, 64, 128], 
                 kernel_size=5, output_size=2):
        super(CNNBaseline, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_size = output_size
        
        # Build convolutional layers
        layers = []
        in_channels = input_channels
        
        for out_channels in hidden_channels:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, 
                         padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(0.3)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, output_size)
        )
        
    def forward(self, x):
        """
        Forward pass through CNN.
        
        Args:
            x: Input tensor (batch_size, input_size) or 
               (batch_size, channels, length)
               
        Returns:
            logits: Class logits (batch_size, output_size)
        """
        # Ensure input has channel dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_size)
        
        # Convolutional feature extraction
        features = self.conv_layers(x)  # (batch_size, channels, length)
        
        # Global pooling
        pooled = self.gap(features).squeeze(-1)  # (batch_size, channels)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits
    
    def count_macs(self, input_size):
        """
        Count multiply-accumulate operations (MACs) for CNN.
        This provides a fair comparison with SNN SOPs.
        
        Args:
            input_size: Input sequence length
            
        Returns:
            total_macs: Total MAC operations
        """
        macs = 0
        length = input_size
        in_channels = self.input_channels
        
        for out_channels in self.hidden_channels:
            # Conv1d MACs: out_channels * in_channels * kernel_size * length
            macs += out_channels * in_channels * 5 * length
            
            # BatchNorm MACs: ~2 * out_channels * length
            macs += 2 * out_channels * length
            
            # Pooling: negligible
            length = length // 2
            in_channels = out_channels
        
        # Classifier MACs
        macs += self.hidden_channels[-1] * 64  # First FC layer
        macs += 64 * self.output_size  # Output layer
        
        return macs
