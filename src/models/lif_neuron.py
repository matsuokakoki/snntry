"""
Leaky Integrate-and-Fire (LIF) neuron implementation with SOPs tracking.
"""

import torch
import torch.nn as nn


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron layer with synaptic operations counting.
    
    Args:
        input_size: Number of input features
        output_size: Number of output neurons
        tau_mem: Membrane time constant
        tau_syn: Synaptic time constant
        threshold: Firing threshold
        dt: Time step
    """
    
    def __init__(self, input_size, output_size, tau_mem=10.0, tau_syn=5.0, 
                 threshold=1.0, dt=1.0):
        super(LIFNeuron, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.threshold = threshold
        self.dt = dt
        
        # Synaptic weights
        self.fc = nn.Linear(input_size, output_size, bias=True)
        
        # Decay factors
        self.alpha = torch.exp(torch.tensor(-dt / tau_mem))
        self.beta = torch.exp(torch.tensor(-dt / tau_syn))
        
        # SOPs counter
        self.sops_count = 0
        
    def reset_states(self, batch_size, device):
        """Reset membrane potential and synaptic current."""
        self.mem = torch.zeros(batch_size, self.output_size, device=device)
        self.syn = torch.zeros(batch_size, self.output_size, device=device)
        self.sops_count = 0
        
    def forward(self, x, track_sops=False):
        """
        Forward pass through LIF neuron.
        
        Args:
            x: Input spike tensor (batch_size, input_size)
            track_sops: Whether to count synaptic operations
            
        Returns:
            spikes: Output spikes (batch_size, output_size)
        """
        # Synaptic input: I = Wx
        syn_input = self.fc(x)
        
        # Track SOPs: each spike multiply-accumulate counts as 1 SOP
        if track_sops:
            # Count only for non-zero inputs (spike-based computation)
            active_synapses = (x.abs() > 0).sum().item()
            self.sops_count += active_synapses * self.output_size
        
        # Update synaptic current: I(t+1) = β*I(t) + syn_input
        self.syn = self.beta * self.syn + syn_input
        
        # Update membrane potential: V(t+1) = α*V(t) + I(t+1)
        self.mem = self.alpha * self.mem + self.syn
        
        # Generate spikes and reset
        spikes = (self.mem >= self.threshold).float()
        self.mem = self.mem * (1.0 - spikes)  # Reset spiked neurons
        
        return spikes
    
    def get_sops(self):
        """Return the accumulated SOPs count."""
        return self.sops_count
    
    def reset_sops(self):
        """Reset SOPs counter."""
        self.sops_count = 0


class SNNLayer(nn.Module):
    """
    SNN layer combining LIF neurons with named tracking.
    """
    
    def __init__(self, input_size, output_size, tau_mem=10.0, tau_syn=5.0, 
                 threshold=1.0, dt=1.0, name="layer"):
        super(SNNLayer, self).__init__()
        
        self.lif = LIFNeuron(input_size, output_size, tau_mem, tau_syn, 
                            threshold, dt)
        self.name = name
        
    def reset_states(self, batch_size, device):
        """Reset neuron states."""
        self.lif.reset_states(batch_size, device)
        
    def forward(self, x, track_sops=False):
        """Forward pass."""
        return self.lif(x, track_sops)
    
    def get_sops(self):
        """Get layer-wise SOPs."""
        return {self.name: self.lif.get_sops()}
    
    def reset_sops(self):
        """Reset SOPs counter."""
        self.lif.reset_sops()
