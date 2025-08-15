from resistor_classification import predict_resistance
import pickle
from tensorflow.keras.models import load_model

# Load model & label encoder
model = load_model('resistor_model.h5')
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Predict
predicted_value, confidence = predict_resistance('path/to/image.jpg', model, label_encoder)
print(f"Predicted: {predicted_value} ({confidence:.2f} confidence)")
