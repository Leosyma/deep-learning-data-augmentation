# 🌼 Flower Image Classification with CNN and Data Augmentation

This project implements a **Convolutional Neural Network (CNN)** for multi-class image classification, enhanced with **data augmentation techniques** to improve model generalization and robustness.

The pipeline uses the **TensorFlow Flowers Dataset** and demonstrates how data augmentation can reduce overfitting and improve performance when working with limited image data.

---

## 📌 Project Overview

- Builds a custom **CNN from scratch** using TensorFlow and Keras  
- Performs **multi-class classification** of flower images  
- Applies **data augmentation** to artificially increase dataset diversity  
- Compares model performance **with and without data augmentation**

The model classifies images into **five flower categories**:
- Roses  
- Daisies  
- Dandelions  
- Sunflowers  
- Tulips  

---

## 🧠 Model Architecture

### Baseline CNN
- Convolutional layers: 16 → 32 → 64 filters
- MaxPooling layers for spatial reduction
- Fully connected Dense layer
- Output layer with 5 classes

### Enhanced CNN with Data Augmentation
- Random horizontal flip
- Random rotation
- Random zoom
- Dropout layer for regularization

---

## 🔄 Data Augmentation Techniques

Data augmentation is applied using Keras preprocessing layers:
- `RandomFlip`
- `RandomRotation`
- `RandomZoom`

These transformations generate new training samples at runtime, improving the model’s ability to generalize to unseen data.

---

## 🗂 Dataset

- **TensorFlow Flowers Dataset**
- Automatically downloaded and extracted
- Images resized to **180 × 180**
- Pixel values normalized to the range **[0, 1]**
- Split into training and testing sets using `train_test_split`

---

## ⚙️ Technologies & Libraries

- Python
- TensorFlow & Keras
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn
- PIL

---

## 🚀 Key Learning Objectives

- Build CNNs from scratch for image classification
- Understand the impact of data augmentation
- Apply regularization techniques such as dropout
- Improve model generalization on small datasets
- Visualize augmented image samples

---

## 📈 Results

The augmented model demonstrates improved generalization compared to the baseline CNN, highlighting the effectiveness of data augmentation techniques in computer vision tasks.

---

## 📄 Notes

This project is intended for educational purposes and serves as a practical example of image classification using CNNs and data augmentation with TensorFlow.
