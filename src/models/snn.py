"""
Temporal Spiking Neural Network for ECG anomaly detection.
"""

import torch
import torch.nn as nn
from .lif_neuron import SNNLayer


class TemporalSNN(nn.Module):
    """
    Multi-layer temporal SNN with layer-wise SOPs tracking.
    
    Args:
        input_size: Input dimension
        hidden_sizes: List of hidden layer sizes
        output_size: Number of output classes
        time_steps: Number of time steps for temporal processing
        tau_mem: Membrane time constant
        tau_syn: Synaptic time constant
        threshold: Firing threshold
        dt: Time step
    """
    
    def __init__(self, input_size, hidden_sizes, output_size, time_steps=100,
                 tau_mem=10.0, tau_syn=5.0, threshold=1.0, dt=1.0):
        super(TemporalSNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.time_steps = time_steps
        
        # Build layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(
            SNNLayer(input_size, hidden_sizes[0], tau_mem, tau_syn, 
                    threshold, dt, name="layer_0")
        )
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(
                SNNLayer(hidden_sizes[i], hidden_sizes[i+1], tau_mem, tau_syn,
                        threshold, dt, name=f"layer_{i+1}")
            )
        
        # Output layer
        self.layers.append(
            SNNLayer(hidden_sizes[-1], output_size, tau_mem, tau_syn,
                    threshold, dt, name="output_layer")
        )
        
        # Input encoding
        self.input_encoder = RateEncoder(time_steps)
        
    def reset_states(self, batch_size, device):
        """Reset all layer states."""
        for layer in self.layers:
            layer.reset_states(batch_size, device)
            
    def reset_sops(self):
        """Reset SOPs counters for all layers."""
        for layer in self.layers:
            layer.reset_sops()
            
    def get_layer_sops(self):
        """Get SOPs breakdown by layer."""
        sops_dict = {}
        for layer in self.layers:
            sops_dict.update(layer.get_sops())
        return sops_dict
    
    def get_total_sops(self):
        """Get total SOPs across all layers."""
        return sum(self.get_layer_sops().values())
    
    def forward(self, x, track_sops=False):
        """
        Forward pass through temporal SNN.
        
        Args:
            x: Input tensor (batch_size, input_size)
            track_sops: Whether to track synaptic operations
            
        Returns:
            output: Spike counts over time (batch_size, output_size)
        """
        batch_size = x.size(0)
        device = x.device
        
        # Reset states for new sequence
        self.reset_states(batch_size, device)
        
        if track_sops:
            self.reset_sops()
        
        # Encode input as spike trains
        spike_trains = self.input_encoder(x)  # (batch_size, time_steps, input_size)
        
        # Accumulate output spikes
        output_spikes = torch.zeros(batch_size, self.output_size, device=device)
        
        # Process each time step
        for t in range(self.time_steps):
            spikes = spike_trains[:, t, :]  # (batch_size, input_size)
            
            # Forward through all layers
            for layer in self.layers:
                spikes = layer(spikes, track_sops=track_sops)
            
            # Accumulate output spikes
            output_spikes += spikes
        
        return output_spikes


class RateEncoder(nn.Module):
    """
    Rate-based encoder: converts continuous values to spike trains.
    Higher values generate more spikes.
    """
    
    def __init__(self, time_steps):
        super(RateEncoder, self).__init__()
        self.time_steps = time_steps
        
    def forward(self, x):
        """
        Encode input as Poisson spike train.
        
        Args:
            x: Input tensor (batch_size, input_size)
            
        Returns:
            spike_train: (batch_size, time_steps, input_size)
        """
        batch_size, input_size = x.size()
        
        # Normalize to [0, 1] range
        x_norm = torch.sigmoid(x)
        
        # Expand to time dimension
        x_expanded = x_norm.unsqueeze(1).repeat(1, self.time_steps, 1)
        
        # Generate Poisson spikes with detached randomness for gradient flow
        # Use straight-through estimator: forward uses spikes, backward uses rates
        rand_vals = torch.rand_like(x_expanded)
        spike_train = (rand_vals < x_expanded).float()
        
        # Straight-through estimator: substitute gradient
        spike_train = spike_train - x_expanded.detach() + x_expanded
        
        return spike_train


class LatencyEncoder(nn.Module):
    """
    Latency-based encoder: higher values spike earlier.
    More biologically plausible for temporal coding.
    """
    
    def __init__(self, time_steps, tau=10.0):
        super(LatencyEncoder, self).__init__()
        self.time_steps = time_steps
        self.tau = tau
        
    def forward(self, x):
        """
        Encode input as latency-coded spikes.
        
        Args:
            x: Input tensor (batch_size, input_size)
            
        Returns:
            spike_train: (batch_size, time_steps, input_size)
        """
        batch_size, input_size = x.size()
        device = x.device
        
        # Normalize to [0, 1]
        x_norm = torch.sigmoid(x)
        
        # Convert to latency: higher values -> earlier spikes
        # latency = -tau * log(x_norm + epsilon)
        epsilon = 1e-7
        latency = -self.tau * torch.log(x_norm + epsilon)
        
        # Clip to time range
        latency = torch.clamp(latency, 0, self.time_steps - 1)
        
        # Create spike train
        spike_train = torch.zeros(batch_size, self.time_steps, input_size, device=device)
        
        # Place spikes at latency times
        for b in range(batch_size):
            for i in range(input_size):
                t = int(latency[b, i].item())
                if t < self.time_steps:
                    spike_train[b, t, i] = 1.0
        
        return spike_train
