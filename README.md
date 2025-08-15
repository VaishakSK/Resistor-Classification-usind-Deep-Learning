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

FINAL_DATASET/ <br>
├── 2K2_4W/<br>
│ ├── image1.jpg<br>
│ ├── image2.jpg<br>
├── 4K7_2W/<br>
│ ├── image1.jpg<br>
│ ├── image2.jpg<br>


**Example:**<br>
- **2K2_4W** → 2.2 kΩ, 4 Watt resistor  <br>
- **4K7_2W** → 4.7 kΩ, 2 Watt resistor  <br>

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

📊 Example Query & Output<br>

Query: Predict the resistor value from an image.<br>
Example Image: FINAL_DATASET/2K2_4W/image2.jpg<br>

Output:
Predicted resistance: 2K2_4W<br>
Confidence: 0.98<br>

📈 Training Visualization
The script generates accuracy and loss plots:<br>
Accuracy: Shows how well the model is learning.<br>
Loss: Measures the model’s error.<br>
