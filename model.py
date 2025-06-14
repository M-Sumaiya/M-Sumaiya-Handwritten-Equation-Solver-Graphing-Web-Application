import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
def load_trained_model():
    model = load_model("digit_recognition_model_new_2.h5")
    return model
