import streamlit as st
from PIL import Image
from deepface import DeepFace
import numpy as np

def verify_faces(image1, image2):
    try:
        # Use DeepFace to verify faces (returns a dictionary with 'verified' key)
        result = DeepFace.verify(image1, image2, model_name='Facenet')
        verified = result["verified"]
        distance = result['distance']
        
        # Convert the distance to a similarity score
        similarity = (1 - distance) * 100  # Normalize distance as similarity score

        return verified, similarity
    except ValueError as e:
        return False, str(e)

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
            verified, similarity = verify_faces(np.array(image1), np.array(image2))
            if isinstance(similarity, str):
                st.error(similarity)
            else:
                if verified:
                    st.success(f"Face Verified with {similarity:.2f}% similarity")
                else:
                    st.error(f"Face Not Verified. Similarity: {similarity:.2f}%")

if __name__ == "__main__":
    main()
