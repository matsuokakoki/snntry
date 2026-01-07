"""
Evaluation metrics for ECG anomaly detection.
"""

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_fscore_support, roc_auc_score
)


def compute_metrics(y_true, y_pred, y_scores=None):
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_scores: Prediction scores (for AUC)
        
    Returns:
        metrics: Dictionary of metrics
    """
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Basic metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall, TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # F1 scores
    f1_normal = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) > 0 else 0
    f1_anomaly = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    macro_f1 = (f1_normal + f1_anomaly) / 2
    
    # Weighted F1
    _, _, _, support = precision_recall_fscore_support(y_true, y_pred)
    weighted_f1 = (f1_normal * support[0] + f1_anomaly * support[1]) / sum(support)
    
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # AUC if scores provided
    auc = None
    if y_scores is not None:
        try:
            auc = roc_auc_score(y_true, y_scores)
        except:
            auc = None
    
    metrics = {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'accuracy': accuracy,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'auc': auc
    }
    
    return metrics


def evaluate_model(model, test_loader, device, model_type='snn', track_sops=False):
    """
    Evaluate model on test set.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Computation device
        model_type: 'snn' or 'cnn'
        track_sops: Whether to track SOPs
        
    Returns:
        metrics: Evaluation metrics
        predictions: Predicted labels
        scores: Prediction scores
    """
    model.eval()
    
    all_labels = []
    all_predictions = []
    all_scores = []
    total_sops = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            if model_type == 'snn':
                output = model(data, track_sops=track_sops)
                if track_sops:
                    total_sops += model.get_total_sops()
            else:
                output = model(data)
            
            # Get predictions
            scores = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            
            all_labels.extend(target.cpu().numpy())
            all_predictions.extend(pred.cpu().numpy())
            all_scores.extend(scores[:, 1].cpu().numpy())  # Score for anomaly class
    
    # Compute metrics
    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_scores)
    )
    
    if track_sops:
        metrics['total_sops'] = total_sops
    
    return metrics, all_predictions, all_scores


def check_performance_targets(metrics, targets):
    """
    Check if metrics meet target thresholds.
    
    Args:
        metrics: Computed metrics
        targets: Target thresholds
        
    Returns:
        meets_targets: Boolean indicating if all targets are met
        report: String report
    """
    meets_sensitivity = metrics['sensitivity'] >= targets['sensitivity']
    meets_specificity = metrics['specificity'] >= targets['specificity']
    meets_f1 = metrics['macro_f1'] >= targets['macro_f1']
    
    meets_targets = meets_sensitivity and meets_specificity and meets_f1
    
    report = f"""
Performance Evaluation:
-----------------------
Sensitivity: {metrics['sensitivity']:.4f} (Target: ≥{targets['sensitivity']}) {'✓' if meets_sensitivity else '✗'}
Specificity: {metrics['specificity']:.4f} (Target: ≥{targets['specificity']}) {'✓' if meets_specificity else '✗'}
Macro F1:    {metrics['macro_f1']:.4f} (Target: ≥{targets['macro_f1']}) {'✓' if meets_f1 else '✗'}
-----------------------
Overall:     {'PASS' if meets_targets else 'FAIL'}
    """
    
    return meets_targets, report


def print_detailed_metrics(metrics, model_name="Model"):
    """
    Print detailed evaluation metrics.
    
    Args:
        metrics: Computed metrics
        model_name: Name of the model
    """
    print(f"\n{model_name} Evaluation Results:")
    print("=" * 50)
    print(f"Sensitivity (Recall): {metrics['sensitivity']:.4f}")
    print(f"Specificity:          {metrics['specificity']:.4f}")
    print(f"Precision:            {metrics['precision']:.4f}")
    print(f"Macro F1:             {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:          {metrics['weighted_f1']:.4f}")
    print(f"Accuracy:             {metrics['accuracy']:.4f}")
    if metrics['auc'] is not None:
        print(f"AUC:                  {metrics['auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {metrics['tn']}, FP: {metrics['fp']}")
    print(f"  FN: {metrics['fn']}, TP: {metrics['tp']}")
    
    if 'total_sops' in metrics:
        print(f"\nSOPs (Synaptic Operations): {metrics['total_sops']}")
    print("=" * 50)
