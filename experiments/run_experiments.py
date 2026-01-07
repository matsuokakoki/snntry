"""
Main experiment script for patient-wise k-fold evaluation.
Compares SNN with CNN baseline and demonstrates SNN advantages.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm

from config import *
from src.models.snn import TemporalSNN
from src.models.cnn_baseline import CNNBaseline
from src.models.stdp import SNNWithSTDP
from src.data.ecg_loader import ECGDataLoader, add_noise_to_data
from src.training.trainer import Trainer
from src.utils.metrics import (
    evaluate_model, check_performance_targets, print_detailed_metrics
)


def run_patient_kfold_experiment(model_type='snn', n_splits=5):
    """
    Run patient-wise k-fold cross-validation.
    
    Args:
        model_type: 'snn' or 'cnn'
        n_splits: Number of folds
        
    Returns:
        results: Dictionary of results across folds
    """
    print(f"\n{'='*60}")
    print(f"Patient-wise {n_splits}-Fold Cross-Validation: {model_type.upper()}")
    print(f"{'='*60}\n")
    
    # Load data
    data_loader = ECGDataLoader(segment_length=SEGMENT_LENGTH)
    data, labels, patient_ids = data_loader.generate_synthetic_data(
        n_patients=20, samples_per_patient=100
    )
    
    # Preprocess
    data = data_loader.preprocess_data(data)
    
    # Create patient-wise folds
    folds = data_loader.create_patient_folds(data, labels, patient_ids, n_splits)
    
    # Store results
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        print(f"\n--- Fold {fold_idx + 1}/{n_splits} ---")
        
        # Create data loaders
        train_loader, test_loader = data_loader.get_dataloaders(
            data, labels, train_idx, test_idx,
            batch_size=TRAINING_CONFIG['batch_size']
        )
        
        # Create model
        if model_type == 'snn':
            model = TemporalSNN(
                input_size=SNN_CONFIG['input_size'],
                hidden_sizes=SNN_CONFIG['hidden_sizes'],
                output_size=SNN_CONFIG['output_size'],
                time_steps=SNN_CONFIG['time_steps'],
                tau_mem=SNN_CONFIG['tau_mem'],
                tau_syn=SNN_CONFIG['tau_syn'],
                threshold=SNN_CONFIG['threshold'],
                dt=SNN_CONFIG['dt']
            )
            track_sops = True
        else:
            model = CNNBaseline(
                input_channels=CNN_CONFIG['input_channels'],
                hidden_channels=CNN_CONFIG['hidden_channels'],
                kernel_size=CNN_CONFIG['kernel_size'],
                output_size=CNN_CONFIG['output_size']
            )
            track_sops = False
        
        # Train
        trainer = Trainer(model, DEVICE, 
                         learning_rate=TRAINING_CONFIG['learning_rate'],
                         model_type=model_type)
        
        history = trainer.train(
            train_loader, test_loader,
            epochs=TRAINING_CONFIG['epochs'],
            early_stopping_patience=TRAINING_CONFIG['early_stopping_patience'],
            track_sops=track_sops
        )
        
        # Evaluate
        metrics, predictions, scores = evaluate_model(
            model, test_loader, DEVICE, 
            model_type=model_type, track_sops=track_sops
        )
        
        fold_results.append(metrics)
        
        # Print fold results
        print_detailed_metrics(metrics, f"Fold {fold_idx + 1}")
    
    # Aggregate results
    aggregate_metrics = {}
    for key in fold_results[0].keys():
        if key not in ['tp', 'tn', 'fp', 'fn', 'total_sops']:
            values = [r[key] for r in fold_results if r[key] is not None]
            if values:
                aggregate_metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
    
    # SOPs aggregation
    if track_sops:
        sops_values = [r['total_sops'] for r in fold_results]
        aggregate_metrics['total_sops'] = {
            'mean': np.mean(sops_values),
            'std': np.std(sops_values)
        }
    
    # Print aggregate results
    print(f"\n{'='*60}")
    print(f"Aggregate Results ({model_type.upper()}):")
    print(f"{'='*60}")
    for key, values in aggregate_metrics.items():
        if key == 'total_sops':
            print(f"{key}: {values['mean']:.0f} ± {values['std']:.0f}")
        else:
            print(f"{key}: {values['mean']:.4f} ± {values['std']:.4f}")
    
    # Check performance targets
    mean_metrics = {k: v['mean'] for k, v in aggregate_metrics.items()}
    meets_targets, report = check_performance_targets(mean_metrics, PERFORMANCE_TARGETS)
    print(report)
    
    return {
        'fold_results': fold_results,
        'aggregate_metrics': aggregate_metrics,
        'meets_targets': meets_targets
    }


def compare_snn_cnn():
    """
    Compare SNN and CNN performance.
    """
    print("\n" + "="*60)
    print("SNN vs CNN COMPARISON")
    print("="*60)
    
    # Run experiments
    snn_results = run_patient_kfold_experiment('snn', n_splits=TRAINING_CONFIG['k_folds'])
    cnn_results = run_patient_kfold_experiment('cnn', n_splits=TRAINING_CONFIG['k_folds'])
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    snn_agg = snn_results['aggregate_metrics']
    cnn_agg = cnn_results['aggregate_metrics']
    
    print(f"\nPerformance Metrics:")
    print(f"{'Metric':<20} {'SNN':<20} {'CNN':<20}")
    print("-" * 60)
    
    for key in ['sensitivity', 'specificity', 'macro_f1', 'accuracy']:
        snn_val = f"{snn_agg[key]['mean']:.4f} ± {snn_agg[key]['std']:.4f}"
        cnn_val = f"{cnn_agg[key]['mean']:.4f} ± {cnn_agg[key]['std']:.4f}"
        print(f"{key:<20} {snn_val:<20} {cnn_val:<20}")
    
    # SOPs comparison
    if 'total_sops' in snn_agg:
        print(f"\nComputational Efficiency:")
        snn_sops = snn_agg['total_sops']['mean']
        print(f"SNN SOPs: {snn_sops:.0f}")
        
        # Estimate CNN MACs
        cnn_model = CNNBaseline()
        cnn_macs = cnn_model.count_macs(SEGMENT_LENGTH)
        print(f"CNN MACs: {cnn_macs:.0f}")
        print(f"Efficiency Ratio (CNN/SNN): {cnn_macs/snn_sops:.2f}x")


def test_noise_robustness():
    """
    Test noise robustness of SNN vs CNN.
    """
    print("\n" + "="*60)
    print("NOISE ROBUSTNESS EVALUATION")
    print("="*60)
    
    # Load data
    data_loader = ECGDataLoader(segment_length=SEGMENT_LENGTH)
    data, labels, patient_ids = data_loader.generate_synthetic_data(
        n_patients=10, samples_per_patient=50
    )
    data = data_loader.preprocess_data(data)
    
    # Single train/test split
    folds = data_loader.create_patient_folds(data, labels, patient_ids, n_splits=2)
    train_idx, test_idx = folds[0]
    
    # Train both models
    print("\nTraining models...")
    
    # SNN
    snn_model = TemporalSNN(
        input_size=SNN_CONFIG['input_size'],
        hidden_sizes=SNN_CONFIG['hidden_sizes'],
        output_size=SNN_CONFIG['output_size'],
        time_steps=SNN_CONFIG['time_steps'],
        tau_mem=SNN_CONFIG['tau_mem'],
        tau_syn=SNN_CONFIG['tau_syn'],
        threshold=SNN_CONFIG['threshold']
    )
    
    train_loader, test_loader = data_loader.get_dataloaders(
        data, labels, train_idx, test_idx, batch_size=32
    )
    
    snn_trainer = Trainer(snn_model, DEVICE, model_type='snn')
    snn_trainer.train(train_loader, test_loader, epochs=20)
    
    # CNN
    cnn_model = CNNBaseline()
    cnn_trainer = Trainer(cnn_model, DEVICE, learning_rate=0.001, model_type='cnn')
    cnn_trainer.train(train_loader, test_loader, epochs=20)
    
    # Test at different noise levels
    print("\nTesting at different noise levels...")
    print(f"{'Noise Level':<15} {'SNN Accuracy':<20} {'CNN Accuracy':<20}")
    print("-" * 55)
    
    for noise_level in NOISE_LEVELS:
        # Add noise
        noisy_data = add_noise_to_data(data, noise_level)
        
        _, noisy_test_loader = data_loader.get_dataloaders(
            noisy_data, labels, train_idx, test_idx, batch_size=32
        )
        
        # Evaluate
        snn_metrics, _, _ = evaluate_model(snn_model, noisy_test_loader, DEVICE, 'snn')
        cnn_metrics, _, _ = evaluate_model(cnn_model, noisy_test_loader, DEVICE, 'cnn')
        
        print(f"{noise_level:<15.2f} {snn_metrics['accuracy']:<20.4f} {cnn_metrics['accuracy']:<20.4f}")


def demonstrate_stdp_adaptation():
    """
    Demonstrate STDP-based online adaptation.
    """
    print("\n" + "="*60)
    print("STDP-BASED ONLINE ADAPTATION")
    print("="*60)
    
    # Load data
    data_loader = ECGDataLoader(segment_length=SEGMENT_LENGTH)
    data, labels, patient_ids = data_loader.generate_synthetic_data(
        n_patients=10, samples_per_patient=50
    )
    data = data_loader.preprocess_data(data)
    
    # Split: train on most patients, adapt on one patient
    unique_patients = np.unique(patient_ids)
    adaptation_patient = unique_patients[-1]
    
    train_mask = patient_ids != adaptation_patient
    adapt_mask = patient_ids == adaptation_patient
    
    train_idx = np.where(train_mask)[0]
    adapt_idx = np.where(adapt_mask)[0]
    
    # Train base model
    print("\nTraining base SNN model...")
    base_snn = TemporalSNN(
        input_size=SNN_CONFIG['input_size'],
        hidden_sizes=SNN_CONFIG['hidden_sizes'],
        output_size=SNN_CONFIG['output_size'],
        time_steps=SNN_CONFIG['time_steps'],
        tau_mem=SNN_CONFIG['tau_mem'],
        tau_syn=SNN_CONFIG['tau_syn']
    )
    
    train_loader, _ = data_loader.get_dataloaders(
        data, labels, train_idx, adapt_idx[:len(adapt_idx)//2], batch_size=32
    )
    
    trainer = Trainer(base_snn, DEVICE, model_type='snn')
    trainer.train(train_loader, _, epochs=20)
    
    # Test before adaptation
    test_data = data[adapt_idx[len(adapt_idx)//2:]]
    test_labels = labels[adapt_idx[len(adapt_idx)//2:]]
    
    from torch.utils.data import DataLoader
    from src.data.ecg_loader import ECGDataset
    test_dataset = ECGDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print("\nPerformance before STDP adaptation:")
    metrics_before, _, _ = evaluate_model(base_snn, test_loader, DEVICE, 'snn')
    print(f"Accuracy: {metrics_before['accuracy']:.4f}")
    print(f"Macro F1: {metrics_before['macro_f1']:.4f}")
    
    # Apply STDP adaptation
    print("\nApplying STDP adaptation...")
    snn_with_stdp = SNNWithSTDP(base_snn, STDP_CONFIG)
    
    adapt_data = data[adapt_idx[:len(adapt_idx)//2]]
    adapt_labels = labels[adapt_idx[:len(adapt_idx)//2]]
    
    for i in range(10):  # Few adaptation steps
        adapt_batch = torch.FloatTensor(adapt_data).to(DEVICE)
        adapt_target = torch.LongTensor(adapt_labels).to(DEVICE)
        snn_with_stdp.adapt_with_stdp(adapt_batch, adapt_target, learning_rate=0.001)
    
    print("\nPerformance after STDP adaptation:")
    metrics_after, _, _ = evaluate_model(snn_with_stdp.base_snn, test_loader, DEVICE, 'snn')
    print(f"Accuracy: {metrics_after['accuracy']:.4f}")
    print(f"Macro F1: {metrics_after['macro_f1']:.4f}")
    
    improvement = (metrics_after['accuracy'] - metrics_before['accuracy']) * 100
    print(f"\nImprovement: {improvement:+.2f}%")


if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("="*60)
    print("TIME-DOMAIN SNN FOR ECG ANOMALY DETECTION")
    print("="*60)
    
    # Main experiments
    print("\n1. Patient-wise K-Fold Evaluation")
    snn_results = run_patient_kfold_experiment('snn', n_splits=TRAINING_CONFIG['k_folds'])
    
    print("\n2. CNN Baseline Comparison")
    compare_snn_cnn()
    
    print("\n3. Noise Robustness Test")
    test_noise_robustness()
    
    print("\n4. STDP-based Adaptation")
    demonstrate_stdp_adaptation()
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*60)
