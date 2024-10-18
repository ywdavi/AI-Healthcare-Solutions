# AI-Healthcare-Solutions

_Authors: Simone Vaccari, Alessio De Luca, Davide Vettore_

This repository contains two projects that utilize machine learning and AI to provide advanced healthcare solutions. The work was completed in the second semester of my Master's program (2023/24). The focus is on enhancing diagnostic capabilities in two domains: epilepsy symptom recognition from EEG signals and breast cancer mass detection through ultrasound images.

## Project 1: Epilepsy Detection via EEG Analysis

### Objectives
- Automate the recognition of epilepsy symptoms from EEG signals.
- Provide neurologists with a reliable tool for diagnosing epilepsy efficiently.
  
### Key Steps
- **Data Preparation**: 
  - Dataset of 500 EEG signals, each containing 4094 measurements.
  - Z-score normalization and outlier detection using two thresholds (1 and 2 standard deviations).
  
- **Machine Learning**:
  - Classified EEG signals into mild and mild-severe epilepsy using Support Vector Machines (SVM) and Random Forests.
  - Hyperparameter tuning using 5-fold cross-validation and grid search.
  
- **Feature Engineering**:
  - Principal Component Analysis (PCA) and K-Best techniques for feature selection.
  
### Results
- The best model (Random Forest with 200 estimators and log-loss criterion) achieved:
  - **Accuracy**: 90.3%
  - **Sensitivity**: 93.7%
  - **Specificity**: 87.0%
- Significant improvement in diagnostic accuracy, especially for severe cases of epilepsy.

---

## Project 2: Breast Cancer Detection via Ultrasound Imaging

### Objectives
- Develop an AI tool to identify and classify breast lumps as cancerous or non-cancerous from ultrasound images.
- Aid oncologists in early and accurate diagnosis of breast cancer.

### Key Steps
- **Image Preprocessing**:
  - Ultrasound images were resized to 128x128, converted to grayscale, and normalized.
  - Data augmentation (horizontal flip and slight rotation) to increase training set size.

- **Image Segmentation**:
  - A U-Net Convolutional Neural Network was used for mass segmentation.
  - The model was trained over 100 epochs using the Adam optimizer, achieving:
    - **Mean IoU**: 75%
    - **DICE Coefficient**: 72%
    - **Precision/Recall/F1**: 72%

- **Classification**:
  - Support Vector Machine (SVM) with a radial basis function kernel was used to classify cancerous and non-cancerous masses.
  - After hyperparameter tuning, the model achieved:
    - **Accuracy**: 97%
    - **Sensitivity**: 95%
    - **Specificity**: 98%
      
- Finally, we design a dashboard with an intuitive interface for diagnostics.

### Results
- The integrated tool significantly enhances breast cancer diagnosis by offering accurate classification and segmentation of breast lumps from ultrasound images.

---

## Conclusion
Both projects demonstrate the potential of machine learning in improving healthcare diagnostics by automating complex tasks, thus enhancing early detection and treatment outcomes for epilepsy and breast cancer patients.

