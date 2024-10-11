import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import gdown

# Download the pre-trained model
@st.cache_resource
def download_model():
    url = 'https://drive.google.com/uc?id=1-HxJOVDzq8Gw9_DkNd1l5ZlTQF_qdHn3'
    output = 'siamese_model.h5'
    gdown.download(url, output, quiet=False)
    return load_model('siamese_model.h5', compile=False)

# Load the face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the Siamese model
model = download_model()

def preprocess_image(image):
    # Convert PIL Image to numpy array
    image = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return None
    
    # Get the first face
    x, y, w, h = faces[0]
    
    # Extract face ROI
    face = image[y:y+h, x:x+w]
    
    # Resize to 224x224 (input size for ResNet50)
    face = cv2.resize(face, (224, 224))
    
    # Preprocess for ResNet50
    face = preprocess_input(face)
    
    return face

def verify_faces(image1, image2):
    # Preprocess both images
    face1 = preprocess_image(image1)
    face2 = preprocess_image(image2)
    
    if face1 is None or face2 is None:
        return "No face found in one or both images."
    
    # Expand dimensions to create a batch
    face1 = np.expand_dims(face1, axis=0)
    face2 = np.expand_dims(face2, axis=0)
    
    # Make prediction
    similarity = model.predict([face1, face2], verbose=0)[0][0]
    
    return float(similarity)

def main():
    st.markdown("<h1 style='text-align: center'><strong>Facial Verification System</strong></h1>", unsafe_allow_html=True)
    
    # File uploader for the second image
    st.markdown("<h4 style='text-align: center'><strong>Upload the anchor image</strong></h4>", unsafe_allow_html=True)
    image2 = st.file_uploader("", type=['jpg', 'jpeg', 'png'])

    # Webcam input for the first image
    st.markdown("<h4 style='text-align: center'><strong>Capture the live image to verify </strong></h4>", unsafe_allow_html=True)
    image1 = st.camera_input(" ")

    if image1 and image2:
        # Display the captured image
        col1, col2 = st.columns(2)
        with col1:
            st.image(image1, caption="Captured Image", use_column_width=True)
        with col2:
            st.image(image2, caption="Uploaded Image", use_column_width=True)

        # Convert the camera input to an image
        image1 = Image.open(image1)
        image2 = Image.open(image2)

        # Perform face verification
        if st.button("Verify Faces"):
            similarity = verify_faces(image1, image2)
            if isinstance(similarity, str):
                st.error(similarity)
            else:
                if similarity > 0.5:  # Adjust this threshold as needed
                    st.success(f"Similarity Score: {similarity * 100:.2f} %")
                    st.markdown('<h2 style="color:green;"> Face Verified!</h2>', unsafe_allow_html=True)
                else:
                    st.error(f"Similarity Score: {similarity * 100:.2f} %")
                    st.markdown('<h2 style="color:red;"> Face Not Verified!</h2>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
