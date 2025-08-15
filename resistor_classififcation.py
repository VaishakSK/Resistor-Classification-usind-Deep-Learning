"""
Resistor Value Classification using Deep Learning
--------------------------------------------------
This script trains a convolutional neural network (CNN) using EfficientNetB0
to classify resistor images based on their resistance values.

Author: Vaishak Kolhar
Date: 2025-01-15
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle

# -----------------------------
# Constants
# -----------------------------
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 50


# -----------------------------
# Data Loading & Preprocessing
# -----------------------------
def load_and_preprocess_data(dataset_path):
    """
    Loads and preprocesses images from the dataset.

    Args:
        dataset_path (str): Path to dataset folder.

    Returns:
        tuple: (images as numpy array, labels as numpy array)
    """
    images = []
    labels = []

    for class_folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, class_folder)
        if os.path.isdir(folder_path):
            for image_file in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_file)

                # Read and preprocess image
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
                img = img / 255.0  # Normalize pixel values

                images.append(img)
                labels.append(class_folder)

    return np.array(images), np.array(labels)


# -----------------------------
# Model Creation
# -----------------------------
def create_model(num_classes):
    """
    Creates the CNN model using EfficientNetB0 as base.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        tf.keras.Model: Compiled model.
    """
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    base_model.trainable = False  # Freeze base layers

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


# -----------------------------
# Training
# -----------------------------
def train_model():
    """
    Trains the CNN model.

    Returns:
        tuple: (model, training history, label encoder)
    """
    dataset_path = "FINAL_DATASET"
    images, labels = load_and_preprocess_data(dataset_path)

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)

    # Convert labels to one-hot encoding
    categorical_labels = tf.keras.utils.to_categorical(encoded_labels)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        images, categorical_labels, test_size=0.2, random_state=42
    )

    # Create model
    model = create_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ])

    # Train
    history = model.fit(
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .map(lambda x, y: (data_augmentation(x), y))
        .batch(BATCH_SIZE),
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
        ]
    )

    # Save model and label encoder
    model.save('resistor_model.h5')
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    return model, history, label_encoder


# -----------------------------
# Prediction
# -----------------------------
def predict_resistance(image_path, model, label_encoder):
    """
    Predicts the resistor class from an image.

    Args:
        image_path (str): Path to resistor image.
        model (tf.keras.Model): Trained model.
        label_encoder (LabelEncoder): Encoder for class labels.

    Returns:
        tuple: (predicted class, confidence score)
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
    confidence = np.max(prediction)

    return predicted_class[0], confidence


# -----------------------------
# Plotting
# -----------------------------
def plot_training_history(history):
    """
    Plots training & validation accuracy and loss.

    Args:
        history (tf.keras.callbacks.History): Training history object.
    """
    plt.figure(figsize=(12, 4))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    model, history, label_encoder = train_model()
    plot_training_history(history)

    # Example prediction
    test_image_path = "C:/path/to/example_image.JPG"
    predicted_value, confidence = predict_resistance(test_image_path, model, label_encoder)
    print(f"Predicted resistance: {predicted_value}")
    print(f"Confidence: {confidence:.2f}")
