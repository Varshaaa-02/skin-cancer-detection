import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Title
st.title("🩺 Melanoma Detection Using Deep Learning")

# Check for GPU availability
gpu_count = len(tf.config.list_physical_devices('GPU'))
st.write(f"🚀 **Number of GPUs Available:** {gpu_count}")

# Load the trained model
@st.cache_resource
def load_trained_model():
    return load_model(r"C:\Users\varsh\Desktop\Melanoma-Detection-using-using-custom-cnn-master\Melanoma-Detection-using-using-custom-cnn-master\skin_cancer_model.h5")
model = load_trained_model()

# Define 9-class labels (update based on your dataset)
class_labels = [
    "Actinic keratosis",
    "Basal cell carcinoma",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanocytic nevus",
    "Melanoma",
    "Squamous cell carcinoma",
    "Vascular lesion",
    "Unknown"
]

# Upload Image
uploaded_file = st.file_uploader("📤 Upload a Skin Lesion Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Load and display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)  # Fixed deprecated parameter

    # Preprocess the image
    img = image.resize((180, 180))  # Match model input size (Check your model's expected input size)
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make Prediction
    try:
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Ensure predicted class is within bounds
        if predicted_class < len(class_labels):
            predicted_label = class_labels[predicted_class]

            # Display result
            st.write(f"### 🏷 Predicted Diagnosis: **{predicted_label}**")

            # Highlight potential risk cases
            if predicted_label in ["Melanoma", "Squamous cell carcinoma", "Basal cell carcinoma"]:
                st.error("⚠️ This lesion appears to be **cancerous**. Please consult a dermatologist.")
            else:
                st.success("✅ This lesion appears to be **non-cancerous**.")

            # Show confidence scores for all 9 classes
            st.write("### 🔍 Prediction Confidence:")
            for i, label in enumerate(class_labels):
                st.write(f"- **{label}:** {prediction[0][i] * 100:.2f}%")  # ✅ FIXED Indentation here

        else:
            st.error("⚠️ Prediction error: Class index out of range!")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")







