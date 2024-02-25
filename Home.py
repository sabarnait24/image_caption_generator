import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load tokenizer
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

# Load model
model = load_model('model_final.h5')

# Load Xception model
xception_model = Xception(include_top=False, pooling="avg")

max_length = 51

def extract_features(filename, model):
    try:
        image = Image.open(filename)

    except:
        st.error("ERROR: Couldn't open image! Make sure the image path and extension are correct")
        return
    image = image.resize((299, 299))
    image = np.array(image)

    # for images that have 4 channels, we convert them into 3 channels
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image / 127.5
    image = image - 1.0
    feature = model.predict(image, verbose=0)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)

        if word is None:
            break
        in_text += ' ' + word

        if word == 'end':
            break
    return in_text

st.title("Image Caption Generator")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png","webp"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    progress_bar = st.progress(0)

    # Extract features
    photo = extract_features(uploaded_file, xception_model)
    progress_bar.progress(50)
    st.write("## Generated Caption:")

    # Generate caption
    caption = generate_desc(photo, max_length)
    progress_bar.progress(100)
    
    caption = caption.split()[1:-1]
    caption[0] = caption[0].capitalize()
    caption = ' '.join(caption) + '.'
    st.write(caption)
