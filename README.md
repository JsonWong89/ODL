# Chest X-Ray Classification Project

A deep learning project for classifying chest X-ray images into three categories: **Normal**, **Pneumonia**, and **Tuberculosis** using transfer learning with EfficientNet B1.

## 📋 Project Overview

This project implements a medical image classification system using PyTorch and a pretrained EfficientNet B1 model. The model is fine-tuned on chest X-ray images to detect and classify respiratory conditions.

## 🏗️ Dataset Structure

```
code/
├── train/              # Training images (80% of original training data)
│   ├── normal/
│   ├── pneumonia/
│   └── tuberculosis/
├── val/                # Validation images (20% of original training data)
│   ├── normal/
│   ├── pneumonia/
│   └── tuberculosis/
├── test/               # Test images
│   ├── normal/
│   ├── pneumonia/
│   └── tuberculosis/
└── data.yaml           # Dataset configuration
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages
- PyTorch
- torchvision
- Pillow
- matplotlib
- numpy
- pandas
- seaborn
- scikit-learn
- tqdm

## 📓 Notebook Files

### `load_image.ipynb`
Main notebook containing the complete pipeline:
- Data loading and preprocessing
- Data augmentation
- Model architecture (EfficientNet B1)
- Training loop with learning rate scheduling
- Comprehensive evaluation metrics
- Custom image prediction

### `check_GPU.ipynb`
Utility notebook to verify GPU availability and configuration.

## 🔧 Model Architecture

### Base Model
- **EfficientNet B1** (pretrained on ImageNet)
- Input size: 224×224×3
- Frozen pretrained layers for transfer learning

### Custom Classifier Head
```python
Sequential(
    Dropout(0.2),
    Linear(in_features, 512),
    ReLU(inplace=True),
    Dropout(0.3),
    Linear(512, num_classes)
)
```

## 🎯 Training Configuration

### Hyperparameters
- **Batch Size**: 64
- **Epochs**: 20
- **Optimizer**: SGD with momentum (0.9)
- **Learning Rate**: 0.001
- **Scheduler**: CosineAnnealingLR
- **Weight Decay**: 1e-4
- **Loss Function**: CrossEntropyLoss

### Data Augmentation
- Resize to 224×224
- Random horizontal flip (p=0.5)
- Color jitter (brightness, contrast, saturation, hue)
- Normalization

## 📊 Usage

### 1. Data Exploration
```python
# The notebook includes cells to:
# - Verify directory structure
# - Visualize class distribution
# - Display sample images
```

### 2. Train the Model
```python
# Run the training cells in load_image.ipynb
# Training progress will be displayed with:
# - Loss curves (train and validation)
# - Accuracy curves
# - Learning rate schedule
```

### 3. Evaluate the Model
```python
# Comprehensive evaluation includes:
# - Confusion matrix
# - Classification report
# - Per-class metrics (Precision, Recall, F1-Score)
# - Prediction visualizations
```

### 4. Predict on Custom Images
```python
from PIL import Image

# Replace with your image path
image_path = "path/to/your/xray.jpg"

predicted_class, confidence = predict_custom_image(
    image_path=image_path,
    model=model_0,
    transform=data_transform,
    class_names=class_names,
    device=device
)
```

## 📈 Model Performance

The notebook includes comprehensive evaluation metrics:
- **Confusion Matrix**: Visual representation of classification results
- **Classification Report**: Precision, recall, F1-score per class
- **Accuracy Curves**: Training and validation accuracy over epochs
- **Loss Curves**: Training and validation loss over epochs

## 🔬 Features

### Data Analysis
- ✅ Class distribution visualization
- ✅ Sample image visualization
- ✅ Data augmentation preview

### Model Training
- ✅ Transfer learning with EfficientNet B1
- ✅ Learning rate scheduling (Cosine Annealing)
- ✅ Early stopping capability
- ✅ Progress tracking with tqdm

### Evaluation
- ✅ Confusion matrix heatmap
- ✅ Per-class metrics visualization
- ✅ Prediction confidence scores
- ✅ Test set predictions with images

### Prediction
- ✅ Custom image prediction
- ✅ Probability distribution visualization
- ✅ Confidence scores

## 🛠️ Hyperparameter Tuning

The project includes documentation for hyperparameter experiments:
- Learning rate: [0.0001, 0.0005, 0.001, 0.005]
- Batch size: [32, 64, 128]
- Optimizers: Adam, SGD with momentum
- Schedulers: CosineAnnealingLR, StepLR, ReduceLROnPlateau
- Weight decay: [1e-5, 1e-4, 1e-3]
- Dropout rates: [0.2, 0.3, 0.5]

## 📁 File Descriptions

| File | Description |
|------|-------------|
| `load_image.ipynb` | Main training and evaluation notebook |
| `check_GPU.ipynb` | GPU verification utility |
| `requirements.txt` | Python package dependencies |
| `data.yaml` | Dataset configuration file |

## 🖥️ Hardware Requirements

### Recommended
- GPU: NVIDIA GPU with 6GB+ VRAM
- RAM: 16GB+
- Storage: 10GB+ for dataset

### Minimum
- CPU: Multi-core processor
- RAM: 8GB
- Storage: 10GB+

## 📝 Notes

- The validation set is created by splitting 20% of the training data
- Images are automatically resized to 224×224 pixels
- The model supports both CPU and GPU training (GPU recommended)
- Random seeds are set for reproducibility

## 🔮 Future Improvements

- [ ] Implement gradient-weighted class activation mapping (Grad-CAM)
- [ ] Add ensemble models for improved accuracy
- [ ] Experiment with other architectures (ResNet, DenseNet)
- [ ] Implement cross-validation
- [ ] Add model checkpointing
- [ ] Create a web interface for predictions

## 📧 Contact

For questions or collaboration, please reach out through the repository issues.

## 📄 License

This project is for educational purposes.

---

**Note**: This is a medical imaging classification project. Always consult with medical professionals for actual diagnosis and treatment decisions.
