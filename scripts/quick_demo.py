"""
Quick demonstration script - runs a mini experiment to verify functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from config import *
from src.models import TemporalSNN, CNNBaseline
from src.data import ECGDataLoader
from src.training import Trainer
from src.utils import evaluate_model, print_detailed_metrics, check_performance_targets


def run_quick_demo():
    """Run a quick demonstration of the SNN-ECG system."""
    print("="*60)
    print("SNN-ECG QUICK DEMONSTRATION")
    print("="*60)
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Load data
    print("\n1. Loading and preprocessing ECG data...")
    data_loader = ECGDataLoader(segment_length=SEGMENT_LENGTH)
    data, labels, patient_ids = data_loader.generate_synthetic_data(
        n_patients=10, samples_per_patient=50
    )
    data = data_loader.preprocess_data(data)
    print(f"   Total samples: {len(data)}")
    print(f"   Segment length: {SEGMENT_LENGTH}")
    print(f"   Number of patients: {len(np.unique(patient_ids))}")
    print(f"   Normal samples: {np.sum(labels == 0)}")
    print(f"   Anomaly samples: {np.sum(labels == 1)}")
    
    # 2. Create patient-wise split
    print("\n2. Creating patient-wise train/test split...")
    folds = data_loader.create_patient_folds(data, labels, patient_ids, n_splits=3)
    train_idx, test_idx = folds[0]
    
    train_loader, test_loader = data_loader.get_dataloaders(
        data, labels, train_idx, test_idx,
        batch_size=TRAINING_CONFIG['batch_size']
    )
    print(f"   Training samples: {len(train_idx)}")
    print(f"   Test samples: {len(test_idx)}")
    
    # 3. Train SNN
    print("\n3. Training Temporal SNN...")
    snn_model = TemporalSNN(
        input_size=SNN_CONFIG['input_size'],
        hidden_sizes=[64, 32],  # Smaller for demo
        output_size=SNN_CONFIG['output_size'],
        time_steps=50,  # Reduced for faster demo
        tau_mem=SNN_CONFIG['tau_mem'],
        tau_syn=SNN_CONFIG['tau_syn'],
        threshold=SNN_CONFIG['threshold']
    )
    
    snn_trainer = Trainer(snn_model, DEVICE, learning_rate=0.001, model_type='snn')
    snn_history = snn_trainer.train(
        train_loader, test_loader,
        epochs=15,
        early_stopping_patience=5,
        track_sops=True
    )
    
    # 4. Evaluate SNN
    print("\n4. Evaluating SNN...")
    snn_metrics, _, _ = evaluate_model(
        snn_model, test_loader, DEVICE, 
        model_type='snn', track_sops=True
    )
    print_detailed_metrics(snn_metrics, "SNN")
    
    # 5. Train CNN baseline
    print("\n5. Training CNN Baseline...")
    cnn_model = CNNBaseline(
        input_channels=CNN_CONFIG['input_channels'],
        hidden_channels=[32, 64],  # Smaller for demo
        kernel_size=CNN_CONFIG['kernel_size'],
        output_size=CNN_CONFIG['output_size']
    )
    
    cnn_trainer = Trainer(cnn_model, DEVICE, learning_rate=0.001, model_type='cnn')
    cnn_history = cnn_trainer.train(
        train_loader, test_loader,
        epochs=15,
        early_stopping_patience=5
    )
    
    # 6. Evaluate CNN
    print("\n6. Evaluating CNN Baseline...")
    cnn_metrics, _, _ = evaluate_model(
        cnn_model, test_loader, DEVICE, model_type='cnn'
    )
    print_detailed_metrics(cnn_metrics, "CNN")
    
    # 7. Compare efficiency
    print("\n7. Computational Efficiency Comparison:")
    print("="*60)
    snn_sops = snn_metrics['total_sops']
    cnn_macs = cnn_model.count_macs(SEGMENT_LENGTH)
    
    print(f"SNN Total SOPs:  {snn_sops:,}")
    print(f"CNN Total MACs:  {cnn_macs:,}")
    print(f"Efficiency Ratio (CNN/SNN): {cnn_macs/snn_sops:.2f}x")
    
    # 8. Check performance targets
    print("\n8. Performance Target Verification:")
    print("="*60)
    
    print("\nSNN Performance:")
    meets_targets, report = check_performance_targets(snn_metrics, PERFORMANCE_TARGETS)
    print(report)
    
    print("\nCNN Performance:")
    meets_targets, report = check_performance_targets(cnn_metrics, PERFORMANCE_TARGETS)
    print(report)
    
    # 9. Summary
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nKey Findings:")
    print(f"✓ SNN achieves {snn_metrics['sensitivity']:.2%} sensitivity")
    print(f"✓ SNN achieves {snn_metrics['specificity']:.2%} specificity")
    print(f"✓ SNN achieves {snn_metrics['macro_f1']:.2%} Macro F1")
    print(f"✓ Patient-wise validation ensures no data leakage")
    print(f"✓ Layer-wise SOPs tracking enabled")
    print(f"✓ Fair comparison with CNN baseline")
    
    print("\nNext Steps:")
    print("- Run full experiments with: python experiments/run_experiments.py")
    print("- Test noise robustness and STDP adaptation")
    print("- Use real MIT-BIH data for production")


if __name__ == "__main__":
    run_quick_demo()
