# 📷 Resistor Classification using Deep Learning

## 📌 Overview
This project leverages **EfficientNetB0** to classify resistor images into their respective resistance values.  
It processes images, trains a deep learning model, and predicts resistance values from new images.

---

## 🚀 Features
- ✅ Transfer Learning with **EfficientNetB0**
- ✅ Image Augmentation for improved generalization
- ✅ Confidence scores for predictions
- ✅ Saves trained model & label encoder for reuse

---

## 📂 Dataset Structure
The dataset must be organized into subfolders, where each folder name represents a resistor class.  
Each folder should contain all images for that resistor type.

FINAL_DATASET/
├── 2K2_4W/
│ ├── image1.jpg
│ ├── image2.jpg
├── 4K7_2W/
│ ├── image1.jpg
│ ├── image2.jpg


**Example:**
- **2K2_4W** → 2.2 kΩ, 4 Watt resistor  
- **4K7_2W** → 4.7 kΩ, 2 Watt resistor  

---

## 🛠 Installation
Install dependencies:
```bash
pip install tensorflow scikit-learn opencv-python matplotlib numpy
```

▶ Usage
Training the Model
Run:
```bash
python resistor_classification.py
```

📊 Example Query & Output

Query: Predict the resistor value from an image.
Example Image: FINAL_DATASET/2K2_4W/image2.jpg
Output:

Predicted resistance: 2K2_4W
Confidence: 0.98

📈 Training Visualization
The script generates accuracy and loss plots:
Accuracy: Shows how well the model is learning.
Loss: Measures the model’s error.
