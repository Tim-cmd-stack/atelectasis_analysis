# Chest X-Ray Classification and Localization

A deep learning project for automated chest X-ray analysis, focusing on Atelectasis detection and localization using Vision Transformers with Semantic Consistency Modules (SCM).

## 🎯 Project Overview

This project implements a weakly-supervised object localization (WSOL) system for chest X-ray analysis with the following capabilities:

- **Classification**: Multi-class classification of chest X-rays into Atelectasis, Other pathologies, and No pathologies
- **Localization**: Generates attention heatmaps and bounding box predictions for pathological regions
- **Medical Image Processing**: Custom DICOM-style preprocessing with CLAHE enhancement

## 🏗️ Architecture

### Model Components

1. **DeiT Backbone**: Data-efficient Image Transformer (DeiT) as the feature extractor
2. **SCM (Semantic Consistency Module)**: Custom attention diffusion blocks for improved localization
3. **WSOL Framework**: Weakly-supervised localization using only image-level labels

### Key Features

- **Activation Diffusion Blocks**: Novel attention refinement mechanism
- **Multi-head Attention Extraction**: Extracts and processes attention maps from the last transformer layer
- **Semantic Map Generation**: Converts patch tokens to spatial feature maps
- **Dual-mode Operation**: Different behavior during training vs. inference

## 📂 Project Structure

```
├── A_making_data_loader_for_classification.py    # Data loading and preprocessing
├── B_training_teacher_classifier.py              # Model training and evaluation
├── D_test_heatmaps_fixed_bbox.py                # Localization evaluation
├── IDE_PY_FILES/
│   ├── A_creating_classification_dataset.py      # Dataset creation from NIH
│   ├── B_deleting_odd_files.py                   # Data cleaning
│   ├── C_deleting_more_odd_files_for_dataset_reduction.py
│   ├── D_reforming_dataset_paths.py              # Dataset restructuring
│   └── E_making_of_balance_in_dataset.py         # Dataset balancing
├── localization_samples/                         # Generated attention maps
├── heatmap_comparisons/                          # Localization results
└── metrics_plots/                                # Training metrics visualization
```

## 🔧 Installation

### Prerequisites

```bash
pip install torch torchvision
pip install timm
pip install opencv-python
pip install scikit-learn
pip install pandas
pip install matplotlib
pip install tqdm
pip install scipy
pip install Pillow
```

### Additional Requirements

- CUDA-compatible GPU (recommended)
- Python 3.8+
- At least 8GB RAM

## 📊 Dataset

The project uses the NIH Chest X-Ray dataset with custom preprocessing:

### Dataset Structure
```
new_classes/
├── Atelectasis/          # Atelectasis cases
├── Other_pathologies/    # Other pathological conditions
└── No_pathologies/       # Normal cases
```

### Data Processing Pipeline

1. **Dataset Creation** (`A_creating_classification_dataset.py`):
   - Processes NIH CSV annotations
   - Categorizes images into 3 classes
   - Copies images to structured folders

2. **Data Cleaning** (`B_deleting_odd_files.py`):
   - Removes mixed-label cases
   - Ensures clean class separation

3. **Dataset Balancing** (`E_making_of_balance_in_dataset.py`):
   - Stratified sampling by age and gender
   - Maintains class proportions
   - Reduces dataset size while preserving diversity

### Custom Preprocessing

The `DicomStyleImagePreprocessor` class applies:
- **Grayscale conversion** for medical image consistency
- **CLAHE enhancement** for contrast improvement
- **Intensity normalization** to 0-255 range
- **RGB channel stacking** for compatibility with pretrained models

## 🚀 Usage

### 1. Data Preparation

```python
# Run dataset creation scripts in order:
python IDE_PY_FILES/A_creating_classification_dataset.py
python IDE_PY_FILES/B_deleting_odd_files.py
python IDE_PY_FILES/C_deleting_more_odd_files_for_dataset_reduction.py
python IDE_PY_FILES/D_reforming_dataset_paths.py
python IDE_PY_FILES/E_making_of_balance_in_dataset.py
```

### 2. Training

```python
# Configure data loader
from A_making_data_loader_for_classification import classification_train_loader, classification_val_loader

# Train the model
python B_training_teacher_classifier.py
```

### 3. Evaluation and Localization

```python
# Generate heatmaps and evaluate localization
python D_test_heatmaps_fixed_bbox.py
```

## 📈 Model Performance

### Training Features

- **Weighted Cross-Entropy Loss** with label smoothing
- **Differential Learning Rates**: Higher LR for SCM components
- **Early Stopping** with patience mechanism
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-score

### Localization Metrics

- **IoU (Intersection over Union)**
- **GIoU (Generalized IoU)**
- **Hausdorff Distance**
- **Average Boundary Distance**

## 🎛️ Configuration

### Key Hyperparameters

```python
# Model Configuration
num_classes = 3
scm_blocks = 4
patch_hw = 14

# Training Configuration
batch_size = 32
num_epochs = 55
backbone_lr = 5e-5
scm_lr = 5e-4
weight_decay = 5e-4

# Class Weights (for imbalanced dataset)
class_weights = [0.7045, 0.1181, 0.1774]
```

### SCM Parameters

- **Lambda (λ)**: Controls diffusion strength (initialized to 2.0)
- **Beta (β)**: Modulates attention refinement (initialized to 0.1)
- **Diffusion Steps**: 4 iterative refinement steps

## 📋 Output Files

### Training Outputs
- `best_deit_scm_model_2.pth`: Best model checkpoint
- `metrics_plots/`: Training curves and validation metrics
- `localization_samples/`: Sample attention visualizations

### Evaluation Outputs
- `heatmap_comparisons/`: Overlay visualizations with bounding boxes
- Quantitative metrics printed to console

## 🔬 Technical Details

### Attention Mechanism

The model extracts attention from the last transformer layer:
```python
# Attention extraction from DeiT
cls2patch = attn_probs[:, :, 0, 1:]  # CLS to patch attention
attention_map = cls2patch.mean(dim=1)  # Average over heads
```

### Semantic Consistency Module

```python
# SCM forward pass
S_flat = F.normalize(S.view(B, C, -1), dim=1)
E = torch.matmul(S_flat.transpose(1, 2), S_flat)  # Similarity matrix
L = (D - A) * (lambda * E - 1)  # Diffusion operator
```

### Localization Pipeline

1. **Attention Extraction**: Get CLS-to-patch attention weights
2. **Semantic Mapping**: Convert patch tokens to spatial features  
3. **SCM Processing**: Apply diffusion-based refinement
4. **Heatmap Generation**: Create localization heatmaps
5. **Bounding Box Extraction**: Convert heatmaps to bounding boxes

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Use gradient accumulation
   - Clear cache with `torch.cuda.empty_cache()`

2. **Poor Localization**:
   - Increase SCM learning rate
   - Adjust lambda and beta parameters
   - Increase number of diffusion steps

3. **Training Instability**:
   - Lower learning rates
   - Increase gradient clipping
   - Use mixed precision training

## 📚 References

- [DeiT: Data-efficient Image Transformers](https://arxiv.org/abs/2012.12877)
- [NIH Chest X-Ray Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data)
- [Weakly-Supervised Object Localization](https://arxiv.org/abs/1506.02025)

## 📄 License

This project is intended for research and educational purposes. Please ensure compliance with the NIH dataset license terms.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add appropriate tests
5. Submit a pull request

## 📧 Contact
For questions or collaborations, please open an issue in the repository.
