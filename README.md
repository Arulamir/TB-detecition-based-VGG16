# Transfer Learning for Tuberculosis Detection Using VGG16

## Overview
This project implements a deep learning approach for Tuberculosis (TB) detection from chest X-ray (CXR) images using transfer learning with the VGG16 architecture. The research findings have been published in the accompanying scientific article, and the full training and fine-tuning implementation is provided in the Python script included in this repository.

The project demonstrates the practical application of convolutional neural networks (CNNs) in medical image classification, with a focus on leveraging pretrained models to improve performance on limited medical datasets.

---

## Repository Contents
- `bu_peni.pdf` — Published research article describing the methodology, experimental setup, and performance evaluation.
- `VGG_DT2_FINE_TUNE_OG.py` — Complete Python implementation for preprocessing, feature extraction, model training, evaluation, and optional fine-tuning using VGG16.

---

## Methodology

### Model Architecture
- Base Model: VGG16 pretrained on ImageNet
- Strategy: Transfer Learning
  - Freeze convolutional base layers
  - Add custom fully connected classifier layers
  - Train classifier layers
  - Optionally unfreeze upper convolutional layers for fine-tuning

### Input Processing
- Image resizing to 224 × 224 × 3
- Pixel normalization (scaled to [0, 1])
- Train-test split (80:20)

### Training Configuration
- Loss Function: Categorical Cross-Entropy
- Optimizer: Adam
- Evaluation Metrics: Accuracy (additional metrics reported in the article)

The final trained model is saved as:

```
tb_classification_vgg16_model.h5
```

---

## Dataset
The study utilizes a combined dataset derived from the Shenzhen and Montgomery chest X-ray datasets (approximately 800 images in total).

For reproducibility, users may obtain publicly available TB chest X-ray datasets (e.g., via Kaggle or official dataset sources) and adjust the image directory path inside `VGG_DT2_FINE_TUNE_OG.py` accordingly.

---

## How to Run

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. Install required dependencies:

```bash
pip install tensorflow numpy scikit-learn matplotlib opencv-python
```

3. Configure the dataset directory inside the script.

4. Run the training script:

```bash
python VGG_DT2_FINE_TUNE_OG.py
```

The script will:
- Load and preprocess images
- Split data into training and testing sets
- Train the classifier with frozen VGG16 layers
- Evaluate performance
- Optionally perform fine-tuning
- Save the trained model

---

## Results
The VGG16-based transfer learning approach achieved competitive classification performance for TB detection on chest X-ray images.

---

## Reproducibility Notes
- Ensure preprocessing during inference matches the training configuration (image size and normalization).
- For systems with limited computational resources, consider using `tensorflow-cpu`.
- Fine-tuning selected convolutional layers may improve performance depending on dataset size and variability.

---

## Author
Muhammad Aamir Nashrullah
- GitHub: https://github.com/Arulamir
- Email: 24051505004@mhs.unesa.ac.id

---

This project highlights practical expertise in deep learning, medical image analysis, transfer learning, and experimental research implementation.
