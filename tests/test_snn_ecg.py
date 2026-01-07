"""
Quick test script to validate the SNN-ECG implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from config import *
from src.models import TemporalSNN, CNNBaseline, SNNWithSTDP
from src.data import ECGDataLoader
from src.training import Trainer
from src.utils import evaluate_model, compute_metrics


def test_data_loading():
    """Test data loading and preprocessing."""
    print("Testing data loading...")
    
    data_loader = ECGDataLoader(segment_length=256)
    data, labels, patient_ids = data_loader.generate_synthetic_data(
        n_patients=5, samples_per_patient=20
    )
    
    assert data.shape[0] == 100, "Expected 100 samples"
    assert data.shape[1] == 256, "Expected segment length 256"
    assert len(np.unique(patient_ids)) == 5, "Expected 5 patients"
    
    print("✓ Data loading works correctly")
    return data, labels, patient_ids


def test_snn_forward():
    """Test SNN forward pass."""
    print("\nTesting SNN forward pass...")
    
    model = TemporalSNN(
        input_size=256,
        hidden_sizes=[64, 32],
        output_size=2,
        time_steps=50
    )
    
    batch_size = 8
    input_data = torch.randn(batch_size, 256)
    
    output = model(input_data)
    
    assert output.shape == (batch_size, 2), f"Expected shape ({batch_size}, 2), got {output.shape}"
    print(f"✓ SNN forward pass works correctly. Output shape: {output.shape}")
    
    # Test SOPs tracking
    output = model(input_data, track_sops=True)
    total_sops = model.get_total_sops()
    layer_sops = model.get_layer_sops()
    
    assert total_sops > 0, "SOPs should be tracked"
    print(f"✓ SOPs tracking works. Total SOPs: {total_sops}")
    print(f"  Layer-wise SOPs: {layer_sops}")


def test_cnn_forward():
    """Test CNN forward pass."""
    print("\nTesting CNN forward pass...")
    
    model = CNNBaseline(
        input_channels=1,
        hidden_channels=[32, 64],
        output_size=2
    )
    
    batch_size = 8
    input_data = torch.randn(batch_size, 256)
    
    output = model(input_data)
    
    assert output.shape == (batch_size, 2), f"Expected shape ({batch_size}, 2), got {output.shape}"
    print(f"✓ CNN forward pass works correctly. Output shape: {output.shape}")
    
    # Test MACs counting
    macs = model.count_macs(256)
    print(f"✓ MACs counting works. Total MACs: {macs}")


def test_training():
    """Test training loop."""
    print("\nTesting training loop...")
    
    # Create small dataset
    data_loader = ECGDataLoader(segment_length=256)
    data, labels, patient_ids = data_loader.generate_synthetic_data(
        n_patients=3, samples_per_patient=20
    )
    data = data_loader.preprocess_data(data)
    
    # Create train/test split
    folds = data_loader.create_patient_folds(data, labels, patient_ids, n_splits=2)
    train_idx, test_idx = folds[0]
    
    train_loader, test_loader = data_loader.get_dataloaders(
        data, labels, train_idx, test_idx, batch_size=8
    )
    
    # Train SNN
    model = TemporalSNN(
        input_size=256,
        hidden_sizes=[32, 16],
        output_size=2,
        time_steps=20  # Reduced for faster testing
    )
    
    trainer = Trainer(model, DEVICE, learning_rate=0.01, model_type='snn')
    history = trainer.train(train_loader, test_loader, epochs=2, 
                           early_stopping_patience=5)
    
    assert len(history['train_loss']) > 0, "Training history should be recorded"
    print(f"✓ Training works. Final train loss: {history['train_loss'][-1]:.4f}")


def test_metrics():
    """Test evaluation metrics."""
    print("\nTesting evaluation metrics...")
    
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    
    metrics = compute_metrics(y_true, y_pred)
    
    assert 'sensitivity' in metrics, "Sensitivity should be computed"
    assert 'specificity' in metrics, "Specificity should be computed"
    assert 'macro_f1' in metrics, "Macro F1 should be computed"
    
    print(f"✓ Metrics computation works correctly")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")


def test_stdp():
    """Test STDP functionality."""
    print("\nTesting STDP...")
    
    base_model = TemporalSNN(
        input_size=256,
        hidden_sizes=[32, 16],
        output_size=2,
        time_steps=20
    )
    
    snn_with_stdp = SNNWithSTDP(base_model, STDP_CONFIG)
    
    batch_size = 4
    input_data = torch.randn(batch_size, 256)
    labels = torch.randint(0, 2, (batch_size,))
    
    # Test adaptation
    snn_with_stdp.enable_stdp()
    output = snn_with_stdp(input_data)
    snn_with_stdp.disable_stdp()
    
    assert output.shape == (batch_size, 2), "STDP forward pass should work"
    print(f"✓ STDP functionality works correctly")


def test_patient_fold():
    """Test patient-wise k-fold splitting."""
    print("\nTesting patient-wise k-fold...")
    
    data_loader = ECGDataLoader(segment_length=256)
    data, labels, patient_ids = data_loader.generate_synthetic_data(
        n_patients=10, samples_per_patient=20
    )
    
    folds = data_loader.create_patient_folds(data, labels, patient_ids, n_splits=5)
    
    assert len(folds) == 5, "Should have 5 folds"
    
    # Check that no patient appears in both train and test
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        train_patients = set(patient_ids[train_idx])
        test_patients = set(patient_ids[test_idx])
        overlap = train_patients & test_patients
        assert len(overlap) == 0, f"Fold {fold_idx} has patient overlap"
    
    print(f"✓ Patient-wise k-fold splitting works correctly")
    print(f"  No patient overlap between train/test in any fold")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("RUNNING SNN-ECG TESTS")
    print("="*60)
    
    try:
        test_data_loading()
        test_snn_forward()
        test_cnn_forward()
        test_metrics()
        test_stdp()
        test_patient_fold()
        test_training()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        return True
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"TEST FAILED ✗")
        print(f"Error: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    success = run_all_tests()
    sys.exit(0 if success else 1)
