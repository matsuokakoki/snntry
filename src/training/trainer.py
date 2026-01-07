"""
Training utilities for SNN and CNN models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np


class Trainer:
    """
    Trainer for both SNN and CNN models.
    """
    
    def __init__(self, model, device, learning_rate=0.001, model_type='snn'):
        self.model = model.to(device)
        self.device = device
        self.model_type = model_type
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader, track_sops=False):
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            track_sops: Whether to track SOPs (SNN only)
            
        Returns:
            avg_loss: Average training loss
            total_sops: Total SOPs if tracked
        """
        self.model.train()
        total_loss = 0
        total_sops = 0
        n_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.model_type == 'snn':
                output = self.model(data, track_sops=track_sops)
                if track_sops:
                    total_sops += self.model.get_total_sops()
            else:
                output = self.model(data)
            
            # Compute loss
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        
        if track_sops and self.model_type == 'snn':
            return avg_loss, total_sops
        else:
            return avg_loss, 0
    
    def validate(self, val_loader, track_sops=False):
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            track_sops: Whether to track SOPs
            
        Returns:
            avg_loss: Average validation loss
            accuracy: Validation accuracy
            total_sops: Total SOPs if tracked
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        total_sops = 0
        n_batches = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                if self.model_type == 'snn':
                    output = self.model(data, track_sops=track_sops)
                    if track_sops:
                        total_sops += self.model.get_total_sops()
                else:
                    output = self.model(data)
                
                # Compute loss
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                # Compute accuracy
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
                n_batches += 1
        
        avg_loss = total_loss / n_batches
        accuracy = correct / total
        
        if track_sops and self.model_type == 'snn':
            return avg_loss, accuracy, total_sops
        else:
            return avg_loss, accuracy, 0
    
    def train(self, train_loader, val_loader, epochs=50, 
             early_stopping_patience=10, track_sops=False):
        """
        Full training loop with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            track_sops: Whether to track SOPs
            
        Returns:
            history: Training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'train_sops': [],
            'val_sops': []
        }
        
        for epoch in range(epochs):
            # Train
            train_loss, train_sops = self.train_epoch(train_loader, track_sops)
            
            # Validate
            val_loss, val_accuracy, val_sops = self.validate(val_loader, track_sops)
            
            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            if track_sops:
                history['train_sops'].append(train_sops)
                history['val_sops'].append(val_sops)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_accuracy:.4f}")
            
            if track_sops and self.model_type == 'snn':
                print(f"  Train SOPs: {train_sops}, Val SOPs: {val_sops}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        
        return history


class EarlyStopping:
    """
    Early stopping helper.
    """
    
    def __init__(self, patience=10, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop
