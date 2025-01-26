import os
import transformers
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import keras
import tensorflow as tf


im = Image.open('/home/jess/Pictures/Screenshots/mush.png')
st.set_page_config(
    page_title="Mush",
    page_icon=im,
    layout="wide",
)


st.title("Mushrooms CDS24")
st.sidebar.title("Sommaire")
pages=["Introduction","Récolte des données","Exploration", "Pre Processing", "Modélisation", "Interprétabilité EfficientNet","Interprétabilité ViT", "Conclusion"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0]:
    st.write('Introduction')
    cover = '/home/jess/Pictures/Screenshots/mush.png'
    st.image(cover)

if page == pages[1]:
    st.write('Récolte des données')

if page == pages[2]:
    st.write('Exploration')

if page == pages[3]:
    st.write('Pre Processing')

if page == pages[4]:
    st.write('Modélisation')

if page == pages[5]:
    st.write('Interpretabilité EfficientNet')

    st.subheader('Prédiction de la classe d\'une image')

    # Choix de l'image à prédire
    choix_eff = ['Image 1', 'Image 2', 'Image 3', 'Image 4']
    option_eff = st.selectbox('Choix de l\'image', choix_eff)
    st.write('L\'image choisie est : ', option_eff)


    def prediction(option_eff):
        if option_eff == 'Image 1':
            img = 'test.png'
            grad = '/home/jess/PycharmProjects/mush_streamlit/imag_pred/eff_test1_gradcam.png'
            sh = '/home/jess/PycharmProjects/mush_streamlit/imag_pred/eff_test1_shap.png'
        elif option_eff == 'Image 2':
            img = 'test2.jpg'
            grad = '/home/jess/PycharmProjects/mush_streamlit/imag_pred/eff_test2_gradcam.png'
            sh = '/home/jess/PycharmProjects/mush_streamlit/imag_pred/eff_test2_shap.png'
        elif option_eff == 'Image 3':
            img = 'test3.jpg'
            grad = '/home/jess/PycharmProjects/mush_streamlit/imag_pred/eff_test3_gradcam.png'
            sh = '/home/jess/PycharmProjects/mush_streamlit/imag_pred/eff_test3_shap.png'
        elif option_eff == 'Image 4':
            img = 'test4.jpg'
            grad = '/home/jess/PycharmProjects/mush_streamlit/imag_pred/eff_test3_gradcam.png'
            sh = '/home/jess/PycharmProjects/mush_streamlit/imag_pred/eff_test3_shap.png'
        return img, grad, sh

    model_eff = tf.keras.models.load_model('/home/jess/PycharmProjects/mush_streamlit/model/tuned_efficientnet_model.keras')
    img, grad, sh = prediction(option_eff)
    image_path = '/home/jess/PycharmProjects/mush_streamlit/test/' + img
    image = Image.open(image_path).resize((224, 224)).convert('RGB')

    predicted_class = np.argmax(model_eff.predict(np.array([image])))
    class_names = sorted(os.listdir('/home/jess/PycharmProjects/mush_streamlit/dataset'))
    predicted_class_name = class_names[predicted_class]
    st.image(image)
    # Print the prediction
    st.write(f"Predicted class: {predicted_class_name}")

    st.subheader('Gradcam Interpretation')
    st.image(grad)

    st.subheader('Shap Interpretation')
    st.image(sh)



if page == pages[6]:
    st.header('Interprétabilité ViT')
    st.subheader('Prédiction de la classe d\'une image')

    # Choix de l'image à prédire
    choix_vit = ['Image 1', 'Image 2', 'Image 3', 'Image 4']
    option_vit = st.selectbox('Choix de l\'image', choix_vit)
    st.write('L\'image choisie est : ', option_vit)

    def prediction(option_vit):
        if option_vit == 'Image 1':
            img = 'test.png'
            sha = '/home/jess/PycharmProjects/mush_streamlit/imag_pred/vit_test1_shap.png'
            cap = '/home/jess/PycharmProjects/mush_streamlit/imag_pred/vit_test1_captum.png'
        elif option_vit == 'Image 2':
            img = 'test2.jpg'
            sha = '/home/jess/PycharmProjects/mush_streamlit/imag_pred/vit_test2_shap.png'
            cap = '/home/jess/PycharmProjects/mush_streamlit/imag_pred/vit_test2_captum.png'
        elif option_vit == 'Image 3':
            img = 'test3.jpg'
            sha = '/home/jess/PycharmProjects/mush_streamlit/imag_pred/vit_test3_shap.png'
            cap = '/home/jess/PycharmProjects/mush_streamlit/imag_pred/vit_test3_captum.png'
        elif option_vit == 'Image 4':
            img = 'test4.jpg'
            sha = '/home/jess/PycharmProjects/mush_streamlit/imag_pred/vit_test4_shap.png'
            cap = '/home/jess/PycharmProjects/mush_streamlit/imag_pred/vit_test4_captum.png'
        return img, sha, cap

    save_path = '/home/jess/PycharmProjects/mush_streamlit/model'
    model_save_path = os.path.join(save_path, "vit_model")

    feature_extractor = transformers.ViTFeatureExtractor.from_pretrained(model_save_path)
    model_vit = transformers.ViTForImageClassification.from_pretrained(model_save_path)
    img, sha, cap = prediction(option_vit)
    image_path = '/home/jess/PycharmProjects/mush_streamlit/test/' + img
    image = Image.open(image_path).resize((224, 224)).convert('RGB')

    # Preprocess the image
    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model_vit(**inputs)
        logits = outputs.logits

    predicted_class_index = logits.argmax(-1).item()

    # Get the predicted class label
    class_names = sorted(os.listdir('/home/jess/PycharmProjects/mush_streamlit/dataset'))
    predicted_class_label = class_names[predicted_class_index]
    predicted_class_name = class_names[predicted_class_index]
    st.image(image)
    # Print the prediction
    st.write(f"Predicted class: {predicted_class_name} ")

    st.subheader('Shap Interpretation')
    st.image(sha)

    st.subheader('Captum Interpretation')
    st.image(cap)

if page == pages[7]:
    st.write('Conclusion')