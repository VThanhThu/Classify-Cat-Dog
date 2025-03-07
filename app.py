import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import base64


st.set_page_config(page_title="Cat Dog Classification", page_icon="🐶🐱", layout="centered")


def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

bg_img_base64 = get_base64_image("D:\\Cat_Dog_Classification\\R.jpg")


page_bg_img = f"""
<style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bg_img_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    h1, h3 {{
        text-align: center;
        color: white;
        text-shadow: 2px 2px 5px black;
    }}
    .uploadedFile {{
        display: flex;
        justify-content: center;
    }}
    .result {{
        text-align: center;
        font-size: 24px;
        color: yellow;
        font-weight: bold;
        text-shadow: 2px 2px 5px black;
    }}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


MODEL_PATH = "D:\\Cat_Dog_Classification\\src\\cat_dog_classifier.keras"
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (128, 128) 

def preprocess_image(img):
    img = img.resize(IMG_SIZE)  
    img_array = image.img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

def predict_image(img):
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    return prediction


st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600&display=swap');

        .title {
            background-color: rgba(0, 0, 0, 0.7);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            color: white;
            font-size: 60px;
            font-family: 'Poppins', sans-serif;
            font-weight: bold;
            text-shadow: 3px 3px 8px black;
            margin-bottom: 15px;
        }

        .subtitle {
            background-color: rgba(0, 0, 0, 0.6);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            color: white;
            font-size: 25px;
            font-family: 'Poppins', sans-serif;
            text-shadow: 2px 2px 5px black;
            margin-bottom: 20px;
        }
    </style>

    <div class="title">🐱 Cat or Dog  🐶</div>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        /* Ẩn thông tin file đã upload */
        div[data-testid="stFileUploader"] div:nth-child(1) {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)


uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    st.image(img, use_column_width=True)

    prediction = predict_image(img)
    class_names = ["Cat 🐱", "Dog 🐶"]  
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    if predicted_class == "Cat 🐱":
        bg_img_base64 = get_base64_image("D:\\Cat_Dog_Classification\\bgcats.png") 
    else:
        bg_img_base64 = get_base64_image("D:\\Cat_Dog_Classification\\bgdog.jpg") 

    dynamic_bg_css = f"""
    <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{bg_img_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
    </style>
    """
    st.markdown(dynamic_bg_css, unsafe_allow_html=True)

    # Hiển thị kết quả
    st.markdown(f"<div class='result'>🔥 Solution: {predicted_class} 🔥</div>", unsafe_allow_html=True)
