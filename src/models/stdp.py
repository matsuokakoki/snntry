"""
Spike-Timing-Dependent Plasticity (STDP) for on-device personalization.
"""

import torch
import torch.nn as nn
import math


class STDP(nn.Module):
    """
    STDP learning rule for synaptic weight adaptation.
    Implements both Long-Term Potentiation (LTP) and Long-Term Depression (LTD).
    
    Args:
        a_plus: LTP amplitude
        a_minus: LTD amplitude
        tau_plus: LTP time constant
        tau_minus: LTD time constant
    """
    
    def __init__(self, a_plus=0.01, a_minus=0.01, tau_plus=20.0, tau_minus=20.0):
        super(STDP, self).__init__()
        
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        
        # Trace decay factors
        self.alpha_plus = math.exp(-1.0 / tau_plus)
        self.alpha_minus = math.exp(-1.0 / tau_minus)
        
        # Spike traces
        self.pre_trace = None
        self.post_trace = None
        
    def reset_traces(self, batch_size, n_pre, n_post, device):
        """Reset spike traces."""
        self.pre_trace = torch.zeros(batch_size, n_pre, device=device)
        self.post_trace = torch.zeros(batch_size, n_post, device=device)
        
    def update(self, pre_spikes, post_spikes, weight):
        """
        Update weights based on STDP rule.
        
        Args:
            pre_spikes: Pre-synaptic spikes (batch_size, n_pre)
            post_spikes: Post-synaptic spikes (batch_size, n_post)
            weight: Current weight matrix (n_post, n_pre)
            
        Returns:
            weight_update: Weight change (n_post, n_pre)
        """
        batch_size = pre_spikes.size(0)
        n_pre = pre_spikes.size(1)
        n_post = post_spikes.size(1)
        device = pre_spikes.device
        
        # Initialize traces if needed
        if self.pre_trace is None:
            self.reset_traces(batch_size, n_pre, n_post, device)
        
        # Update traces: trace(t) = alpha * trace(t-1) + spike(t)
        self.pre_trace = self.alpha_minus * self.pre_trace + pre_spikes
        self.post_trace = self.alpha_plus * self.post_trace + post_spikes
        
        # LTP: post spike causes potentiation based on pre-trace
        # dW = a_plus * post_spike * pre_trace
        ltp = torch.einsum('bi,bj->ij', post_spikes, self.pre_trace) / batch_size
        ltp = self.a_plus * ltp
        
        # LTD: pre spike causes depression based on post-trace
        # dW = -a_minus * pre_spike * post_trace
        ltd = torch.einsum('bj,bi->ij', pre_spikes, self.post_trace) / batch_size
        ltd = -self.a_minus * ltd
        
        # Total weight change
        weight_update = ltp + ltd
        
        return weight_update


class STDPLayer(nn.Module):
    """
    Neural layer with STDP learning capability.
    """
    
    def __init__(self, input_size, output_size, a_plus=0.01, a_minus=0.01,
                 tau_plus=20.0, tau_minus=20.0):
        super(STDPLayer, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        # Weight matrix
        self.weight = nn.Parameter(torch.randn(output_size, input_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(output_size))
        
        # STDP rule
        self.stdp = STDP(a_plus, a_minus, tau_plus, tau_minus)
        
    def forward(self, x):
        """Forward pass."""
        return torch.matmul(x, self.weight.t()) + self.bias
    
    def apply_stdp(self, pre_spikes, post_spikes, learning_rate=1.0):
        """
        Apply STDP weight update.
        
        Args:
            pre_spikes: Pre-synaptic spikes
            post_spikes: Post-synaptic spikes
            learning_rate: Global learning rate multiplier
        """
        with torch.no_grad():
            weight_update = self.stdp.update(pre_spikes, post_spikes, self.weight)
            self.weight.add_(learning_rate * weight_update)
            
            # Optional: weight normalization to prevent unbounded growth
            self.weight.clamp_(-1.0, 1.0)


class SNNWithSTDP(nn.Module):
    """
    SNN with STDP-based online adaptation capability.
    Allows personalization to individual patient data.
    """
    
    def __init__(self, base_snn, stdp_config):
        super(SNNWithSTDP, self).__init__()
        
        self.base_snn = base_snn
        self.stdp_config = stdp_config
        
        # Store spike history for STDP
        self.spike_history = []
        self.stdp_enabled = False
        
    def enable_stdp(self):
        """Enable STDP learning."""
        self.stdp_enabled = True
        
    def disable_stdp(self):
        """Disable STDP learning."""
        self.stdp_enabled = False
        
    def forward(self, x, track_sops=False):
        """
        Forward pass with optional STDP recording.
        """
        # Clear spike history
        self.spike_history = []
        
        batch_size = x.size(0)
        device = x.device
        
        # Reset states
        self.base_snn.reset_states(batch_size, device)
        
        if track_sops:
            self.base_snn.reset_sops()
        
        # Encode input
        spike_trains = self.base_snn.input_encoder(x)
        
        # Accumulate output
        output_spikes = torch.zeros(batch_size, self.base_snn.output_size, device=device)
        
        # Process each time step
        for t in range(self.base_snn.time_steps):
            spikes = spike_trains[:, t, :]
            
            # Store input spikes if STDP enabled
            if self.stdp_enabled:
                layer_spikes = [spikes]
            
            # Forward through layers
            for layer in self.base_snn.layers:
                spikes = layer(spikes, track_sops=track_sops)
                
                if self.stdp_enabled:
                    layer_spikes.append(spikes)
            
            # Store spike history
            if self.stdp_enabled:
                self.spike_history.append(layer_spikes)
            
            output_spikes += spikes
        
        return output_spikes
    
    def adapt_with_stdp(self, x, y, learning_rate=0.01):
        """
        Perform STDP-based adaptation on new patient data.
        
        Args:
            x: Input data
            y: Target labels (for supervision signal)
            learning_rate: STDP learning rate
        """
        self.enable_stdp()
        
        # Forward pass to collect spikes
        output = self.forward(x)
        
        # Apply STDP updates (simplified supervised version)
        # In practice, this could be unsupervised or reward-modulated
        for t, layer_spikes in enumerate(self.spike_history):
            for i in range(len(self.base_snn.layers)):
                layer = self.base_snn.layers[i]
                
                if hasattr(layer, 'lif') and hasattr(layer.lif, 'fc'):
                    pre_spikes = layer_spikes[i]
                    post_spikes = layer_spikes[i + 1]
                    
                    # Simple STDP update (can be enhanced with reward modulation)
                    with torch.no_grad():
                        stdp_rule = STDP(
                            self.stdp_config['a_plus'],
                            self.stdp_config['a_minus'],
                            self.stdp_config['tau_plus'],
                            self.stdp_config['tau_minus']
                        )
                        
                        weight_update = stdp_rule.update(
                            pre_spikes, post_spikes, layer.lif.fc.weight
                        )
                        
                        layer.lif.fc.weight.add_(learning_rate * weight_update)
        
        self.disable_stdp()
        
        return output
