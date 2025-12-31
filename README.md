# 🩺 Deep Learning–Based Pneumonia Detection from Chest X-Ray Images

## Overview
This project applies deep learning techniques to detect pneumonia from chest X-ray images. It follows a progressive model development strategy, starting from a simple CNN trained from scratch and advancing to powerful transfer learning architectures. The focus is on building an accurate, reliable, and interpretable medical image classification system without unnecessary complexity.

---

## Model Evolution
The models were developed and evaluated in a step-by-step manner:
- **Baseline CNN (from scratch):** Established reference performance and identified learning limitations
- **VGG16:** Improved feature extraction using transfer learning
- **ResNet50:** Enabled deeper learning through residual connections
- **DenseNet121:** Achieved the best performance through effective feature reuse

**Best Model:** DenseNet121  
**Accuracy Achieved:** ~97%

---

## Evaluation Strategy
To ensure reliable and unbiased performance, models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC–AUC

Generalization was improved using **data augmentation, hyperparameter tuning, and 5-fold cross-validation** across all models.

---

## Visualization & Interpretability
Model behavior and medical relevance were validated using:
- Training and validation accuracy/loss curves
- Confusion matrices and ROC curves
- **Grad-CAM heatmaps** to highlight lung regions influencing predictions

These visualizations help ensure that predictions are based on clinically meaningful areas.

---

## Tools & Technologies
- **Programming Language:** Python
- **Framework:** TensorFlow / Keras
- **Libraries:** NumPy, Matplotlib, Seaborn

---

## Project Structure
```text

Pneumonia-Detection/
├── data/ 
├── notebooks/ 
├── utils/ 
├── models/
└── README.md
