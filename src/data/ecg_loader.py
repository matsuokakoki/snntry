"""
ECG data loader for MIT-BIH Arrhythmia Database.
Supports patient-wise k-fold cross-validation.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class ECGDataset(Dataset):
    """
    ECG Dataset for anomaly detection.
    
    Args:
        data: ECG segments (n_samples, segment_length)
        labels: Binary labels (n_samples,)
        transform: Optional data transformation
    """
    
    def __init__(self, data, labels, transform=None):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


class ECGDataLoader:
    """
    Data loader with patient-wise k-fold splitting.
    """
    
    def __init__(self, segment_length=256, overlap=0.5):
        self.segment_length = segment_length
        self.overlap = overlap
        self.scaler = StandardScaler()
        
    def generate_synthetic_data(self, n_patients=20, samples_per_patient=100):
        """
        Generate synthetic ECG data for demonstration.
        In practice, replace with MIT-BIH data loading.
        
        Args:
            n_patients: Number of patients
            samples_per_patient: Samples per patient
            
        Returns:
            data, labels, patient_ids
        """
        np.random.seed(42)
        
        all_data = []
        all_labels = []
        all_patient_ids = []
        
        for patient_id in range(n_patients):
            # Normal ECG: smooth sinusoidal pattern
            t = np.linspace(0, 4 * np.pi, self.segment_length)
            normal_samples = samples_per_patient // 2
            
            for _ in range(normal_samples):
                # QRS complex simulation
                signal = 0.5 * np.sin(t) + 0.3 * np.sin(2 * t) + 0.1 * np.sin(3 * t)
                signal += np.random.normal(0, 0.05, self.segment_length)
                
                all_data.append(signal)
                all_labels.append(0)  # Normal
                all_patient_ids.append(patient_id)
            
            # Anomalous ECG: irregular patterns
            anomaly_samples = samples_per_patient - normal_samples
            
            for _ in range(anomaly_samples):
                # Arrhythmia simulation: irregular rhythm, spikes
                signal = 0.5 * np.sin(t * np.random.uniform(0.8, 1.2))
                signal += 0.3 * np.sin(2 * t * np.random.uniform(0.8, 1.2))
                
                # Add random spikes (PVC, etc.)
                spike_positions = np.random.choice(self.segment_length, 
                                                  size=np.random.randint(1, 4))
                signal[spike_positions] += np.random.uniform(0.5, 1.5, len(spike_positions))
                
                signal += np.random.normal(0, 0.1, self.segment_length)
                
                all_data.append(signal)
                all_labels.append(1)  # Anomaly
                all_patient_ids.append(patient_id)
        
        data = np.array(all_data)
        labels = np.array(all_labels)
        patient_ids = np.array(all_patient_ids)
        
        return data, labels, patient_ids
    
    def load_mitbih_data(self, data_dir):
        """
        Load MIT-BIH Arrhythmia Database.
        
        Note: Currently returns synthetic data as a placeholder.
        In practice, use wfdb library:
        - wfdb.rdrecord() to read ECG signals
        - wfdb.rdann() to read annotations
        - Process annotations to create binary labels (normal vs anomaly)
        
        Args:
            data_dir: Directory containing MIT-BIH data
            
        Returns:
            data, labels, patient_ids
        """
        # Placeholder - implement actual MIT-BIH loading
        # For now, return synthetic data
        return self.generate_synthetic_data()
    
    def preprocess_data(self, data):
        """
        Preprocess ECG segments.
        
        Args:
            data: Raw ECG segments
            
        Returns:
            preprocessed_data: Normalized segments
        """
        # Standardization
        n_samples, segment_length = data.shape
        data_flat = data.reshape(-1, 1)
        data_normalized = self.scaler.fit_transform(data_flat)
        data_normalized = data_normalized.reshape(n_samples, segment_length)
        
        return data_normalized
    
    def create_patient_folds(self, data, labels, patient_ids, n_splits=5):
        """
        Create patient-wise k-fold splits.
        Ensures no patient data appears in both train and test.
        
        Args:
            data: ECG segments
            labels: Labels
            patient_ids: Patient IDs for each sample
            n_splits: Number of folds
            
        Returns:
            folds: List of (train_idx, test_idx) tuples
        """
        gkf = GroupKFold(n_splits=n_splits)
        folds = []
        
        for train_idx, test_idx in gkf.split(data, labels, groups=patient_ids):
            folds.append((train_idx, test_idx))
        
        return folds
    
    def get_dataloaders(self, data, labels, train_idx, test_idx, 
                       batch_size=32, shuffle=True):
        """
        Create train and test dataloaders.
        
        Args:
            data: All data
            labels: All labels
            train_idx: Training indices
            test_idx: Test indices
            batch_size: Batch size
            shuffle: Whether to shuffle training data
            
        Returns:
            train_loader, test_loader
        """
        train_data = data[train_idx]
        train_labels = labels[train_idx]
        test_data = data[test_idx]
        test_labels = labels[test_idx]
        
        train_dataset = ECGDataset(train_data, train_labels)
        test_dataset = ECGDataset(test_data, test_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=shuffle, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                shuffle=False, num_workers=0)
        
        return train_loader, test_loader


class NoiseAugmentation:
    """
    Add noise to ECG signals for robustness testing.
    """
    
    def __init__(self, noise_level=0.1):
        self.noise_level = noise_level
        
    def __call__(self, x):
        """Add Gaussian noise to signal."""
        noise = torch.randn_like(x) * self.noise_level
        return x + noise


def add_noise_to_data(data, noise_level=0.1):
    """
    Add noise to test data for robustness evaluation.
    
    Args:
        data: Clean data
        noise_level: Standard deviation of Gaussian noise
        
    Returns:
        noisy_data: Data with added noise
    """
    noise = np.random.normal(0, noise_level, data.shape)
    noisy_data = data + noise
    return noisy_data
