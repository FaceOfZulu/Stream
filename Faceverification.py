import streamlit as st
import cv2
import numpy as np
from PIL import Image

def face_similarity_check(image1, image2):
    # Convert PIL Images to numpy arrays
    image1 = np.array(image1)
    image2 = np.array(image2)

    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    # Load the face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces
    faces1 = face_cascade.detectMultiScale(gray1, 1.1, 4)
    faces2 = face_cascade.detectMultiScale(gray2, 1.1, 4)

    if len(faces1) == 0 or len(faces2) == 0:
        return "No face found in one or both images."

    # Get the first face from each image
    x1, y1, w1, h1 = faces1[0]
    x2, y2, w2, h2 = faces2[0]

    # Extract face ROIs
    face1 = gray1[y1:y1+h1, x1:x1+w1]
    face2 = gray2[y2:y2+h2, x2:x2+w2]

    # Resize faces to the same size
    face1 = cv2.resize(face1, (100, 100))
    face2 = cv2.resize(face2, (100, 100))

    # Compare faces using Mean Squared Error (MSE)
    mse = np.mean((face1 - face2) ** 2)
    similarity = 1 / (1 + mse)  # Convert MSE to a similarity score

    return similarity

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

        # Perform face similarity check
        if st.button("Check Similarity"):
            similarity = face_similarity_check(image1, image2)
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
