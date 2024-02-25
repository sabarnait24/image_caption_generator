import streamlit as st

def main():
    st.title("About Image Caption Generator")
    st.write("Our image caption generator utilizes cutting-edge deep learning techniques, leveraging the powerful Inception model to extract features from images and a trained model to generate descriptive captions.")
    
    st.subheader("Technologies Used:")
    st.write("- TensorFlow: The backbone of our deep learning architecture, providing the framework for training and deploying our models.")
    st.write("- Python: The primary programming language for implementing our image caption generator and integrating various components.")
    st.write("- Inception Model: A pre-trained convolutional neural network (CNN) used to extract meaningful features from input images.")
    st.write("- Natural Language Processing (NLP): Techniques from the field of NLP are employed to generate coherent and contextually relevant captions for the images.")
    
    st.subheader("Project URL:")
    st.write("[https://github.com/sabarnait24/image_caption_generator](https://github.com/sabarnait24/image_caption_generator) - Created by [Sabarna Bhowmik]")
    
    # st.write("Our team is dedicated to pushing the boundaries of computer vision and natural language understanding, aiming to provide accurate and insightful captions for a wide range of images. Join us on our journey to make AI-driven image understanding more accessible and intuitive.")

if __name__ == "__main__":
    main()
