# snn-ecg

Time-domain Spiking Neural Network (SNN) for low-power ECG anomaly detection with SOPs analysis and STDP-based personalization.

## Overview

This repository implements a temporal SNN for ECG anomaly detection with the following key features:

- **Temporal SNN modeling** with Leaky Integrate-and-Fire (LIF) neurons
- **Patient-wise k-fold cross-validation** for robust evaluation
- **Layer-wise Synaptic Operations (SOPs) measurement** for efficiency analysis
- **CNN baseline** for fair comparison
- **STDP-based online learning** for on-device personalization
- **Noise robustness evaluation** to demonstrate SNN advantages

## Performance Targets

- Sensitivity ≥ 0.80
- Specificity ≥ 0.90
- Macro F1 ≥ 0.75

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
snn-ecg/
├── config.py              # Configuration and hyperparameters
├── requirements.txt       # Python dependencies
├── src/
│   ├── models/
│   │   ├── lif_neuron.py      # LIF neuron implementation
│   │   ├── snn.py             # Temporal SNN architecture
│   │   ├── cnn_baseline.py    # CNN baseline for comparison
│   │   └── stdp.py            # STDP learning rule
│   ├── data/
│   │   └── ecg_loader.py      # ECG data loading and preprocessing
│   ├── training/
│   │   └── trainer.py         # Training utilities
│   └── utils/
│       └── metrics.py         # Evaluation metrics
└── experiments/
    └── run_experiments.py     # Main experiment script
```

## Usage

### Run Full Experiments

Run all experiments including k-fold evaluation, CNN comparison, noise robustness, and STDP adaptation:

```bash
python experiments/run_experiments.py
```

### Individual Components

```python
from src.models import TemporalSNN, CNNBaseline, SNNWithSTDP
from src.data import ECGDataLoader
from src.training import Trainer
from src.utils import evaluate_model

# Load data
data_loader = ECGDataLoader(segment_length=256)
data, labels, patient_ids = data_loader.generate_synthetic_data()

# Create SNN model
model = TemporalSNN(
    input_size=256,
    hidden_sizes=[128, 64, 32],
    output_size=2,
    time_steps=100
)

# Train and evaluate
trainer = Trainer(model, device='cuda', model_type='snn')
trainer.train(train_loader, val_loader, epochs=50)
metrics, _, _ = evaluate_model(model, test_loader, device='cuda', model_type='snn')
```

## Key Features

### 1. Patient-wise K-Fold Cross-Validation

Ensures no patient data appears in both training and testing sets, preventing data leakage and providing realistic performance estimates.

```python
from src.data import ECGDataLoader

data_loader = ECGDataLoader()
folds = data_loader.create_patient_folds(data, labels, patient_ids, n_splits=5)
```

### 2. Layer-wise SOPs Tracking

Tracks synaptic operations for energy efficiency analysis:

```python
model = TemporalSNN(...)
output = model(input_data, track_sops=True)
layer_sops = model.get_layer_sops()  # SOPs breakdown by layer
total_sops = model.get_total_sops()  # Total SOPs
```

### 3. STDP-based Online Learning

Enables personalization to individual patient patterns:

```python
from src.models import SNNWithSTDP

snn_with_stdp = SNNWithSTDP(base_model, stdp_config)
snn_with_stdp.adapt_with_stdp(patient_data, patient_labels)
```

### 4. Noise Robustness Testing

Evaluates model performance under various noise conditions:

```python
from src.data import add_noise_to_data

noisy_data = add_noise_to_data(clean_data, noise_level=0.1)
```

## Model Architecture

### Temporal SNN

- **Input encoding**: Rate-based or latency-based spike encoding
- **Hidden layers**: Multiple LIF neuron layers with configurable sizes
- **Time steps**: Temporal processing over 100 time steps
- **Neuronal dynamics**: Membrane and synaptic time constants (τ_mem, τ_syn)

### CNN Baseline

- **Architecture**: 1D convolutional layers with batch normalization
- **Comparable complexity**: Similar parameter count to SNN for fair comparison
- **MAC counting**: Multiply-accumulate operations tracked for efficiency comparison

## Experiments

The main experiment script (`experiments/run_experiments.py`) runs four key evaluations:

1. **Patient-wise K-Fold Evaluation**: 5-fold cross-validation with patient-level splitting
2. **SNN vs CNN Comparison**: Performance and efficiency comparison
3. **Noise Robustness**: Testing at multiple noise levels (0.0 to 0.2)
4. **STDP Adaptation**: Demonstrating online learning capability

## Results

Results are printed to console including:
- Performance metrics (Sensitivity, Specificity, Macro F1)
- Confusion matrices
- SOPs/MACs comparison
- Noise robustness curves
- STDP adaptation improvements

## Configuration

Edit `config.py` to adjust:
- SNN architecture parameters (hidden sizes, time constants, threshold)
- CNN architecture parameters
- Training hyperparameters (learning rate, batch size, epochs)
- STDP parameters
- Performance targets

## Data

The implementation includes synthetic ECG data generation for demonstration. To use real MIT-BIH Arrhythmia Database:

1. Download MIT-BIH database from PhysioNet
2. Implement `load_mitbih_data()` in `src/data/ecg_loader.py` using the `wfdb` library
3. Process annotations to create binary labels (normal vs anomaly)

## Citation

If you use this code, please cite:

```
@software{snn_ecg_2026,
  title={Time-domain Spiking Neural Network for ECG Anomaly Detection},
  author={Your Name},
  year={2026},
  url={https://github.com/matsuokakoki/snn-ecg}
}
```

Note: Update the author name and year when publishing.

## License

See LICENSE file for details.
