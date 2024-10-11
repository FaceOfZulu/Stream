import streamlit as st
import face_recognition
import numpy as np
from PIL import Image
import io

def face_similarity_check(image1, image2):
    # Convert Streamlit images to numpy arrays
    image1 = np.array(image1)
    image2 = np.array(image2)

    # Find face locations
    face_locations1 = face_recognition.face_locations(image1)
    face_locations2 = face_recognition.face_locations(image2)

    if not face_locations1 or not face_locations2:
        return "No face found in one or both images."

    # Get face encodings
    face_encoding1 = face_recognition.face_encodings(image1, face_locations1)[0]
    face_encoding2 = face_recognition.face_encodings(image2, face_locations2)[0]

    # Compare faces
    face_distance = face_recognition.face_distance([face_encoding1], face_encoding2)[0]
    similarity = 1 - face_distance

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
                
                if similarity > 0.7:
                    st.success(f"Similarity Score: {similarity * 100:.2f} %")
                    st.markdown('<h2 style="color:green;"> Face Verified!</h2>', unsafe_allow_html=True)
                else:
                    st.error(f"Similarity Score: {similarity * 100:.2f} %")
                    st.markdown('<h2 style="color:red;"> Face Not Verified!</h2>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
