import os
import transformers
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
#import keras
import tensorflow as tf
import streamlit_extras
#from streamlit_image_viewer import image_viewer
from streamlit_extras.app_logo import add_logo
import streamlit_image_select
from streamlit_image_select import image_select

test_path = 'test/'
pred_path = 'imag_pred/'
model_path = 'model/'
data_path = 'data/'

class_names = {0:"Amanita muscaria",
                1:"Artomyces pyxidatus",
                2:"Boletus edulis",
                3:"Cantharellus cinnabarinus",
                4:"Coprinus comatus",
                5:"Fuligo septica",
                6:"Ganoderma applanatum",
                7:"Ganoderma oregonense",
                8:"Grifola frondosa",
                9:"Hypomyces lactifluorum",
                10:"Lactarius indigo",
                11:"Pluteus cervinus",
                12:"Trametes versicolor"}

#Set favicon & title
im = Image.open('data/mush.png')
st.set_page_config(
    page_title="Mush",
    page_icon=im,
    layout="wide",
)




st.title("Mushrooms CDS24")
st.sidebar.title("Sommaire")
add_logo('data/mush.png')

pages=["Introduction","Récolte des données","Exploration", "Pre Processing", "Modélisation", "Interprétabilité EfficientNet","Interprétabilité ViT", "Conclusion"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0]:
    st.write('Introduction')
    st.image(im)

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


    option = image_select('Choisissez une image de champignon pour connaitre son éspèce :', [(test_path + 'test.png'), (test_path + 'test2.jpg'), (test_path + 'test3.jpg'), (test_path + 'test4.jpg')])
    img = Image.open(option).resize((224, 224)).convert('RGB')
    col1, col2, col3 = st.columns([3,2,3])
    col2.image(img, use_column_width=True, caption='Image à prédire')

    left, right = st.columns(2)
    if left.button("EfficientNet Model", use_container_width=True):
        def prediction(option):
            if option == (test_path + 'test.png'):
                grad = pred_path + 'eff_test1_gradcam.png'
                sh = pred_path + 'eff_test1_shap.png'
            elif option == (test_path + 'test2.jpg'):
                grad = pred_path + 'eff_test2_gradcam.png'
                sh = pred_path + 'eff_test2_shap.png'
            elif option == (test_path + 'test3.jpg'):
                grad = pred_path + 'eff_test3_gradcam.png'
                sh = pred_path + 'eff_test3_shap.png'
            elif option == (test_path + 'test4.jpg'):
                grad = pred_path + 'eff_test4_gradcam.png'
                sh = pred_path + 'eff_test4_shap.png'
            return grad, sh

        model_eff = tf.keras.models.load_model(model_path + 'tuned_efficientnet_model.keras')
        grad, sh = prediction(option)

        predicted_class = np.argmax(model_eff.predict(np.array([img])))
        predicted_class_name = class_names[predicted_class]
        # Print the prediction
        st.write(f"Ce champignon est de l'éspèce {predicted_class_name}")

        st.subheader('Gradcam Interpretation')
        st.image(grad)

        st.subheader('Shap Interpretation')
        col1, col2, col3 = st.columns([0.1, 4, 0.1])
        col2.image(sh, use_column_width=True)

    if right.button("Transformer Model", use_container_width=True):
        def prediction(option):
            if option == (test_path + 'test.png'):
                sha = pred_path + 'vit_test1_shap.png'
                cap = pred_path + 'vit_test1_captum.png'
            elif option == (test_path + 'test2.jpg'):
                sha = pred_path + 'vit_test2_shap.png'
                cap = pred_path + 'vit_test2_captum.png'
            elif option == (test_path + 'test3.jpg'):
                sha = pred_path + 'vit_test3_shap.png'
                cap = pred_path + 'vit_test3_captum.png'
            elif option == (test_path + 'test4.jpg'):
                sha = pred_path + 'vit_test4_shap.png'
                cap = pred_path + 'vit_test4_captum.png'
            return sha, cap

        # Load model
        model_save_path = os.path.join(model_path, "vit_model")
        feature_extractor = transformers.ViTFeatureExtractor.from_pretrained(model_save_path)
        model_vit = transformers.ViTForImageClassification.from_pretrained(model_save_path)
        sha, cap = prediction(option)

        # Preprocess the image
        inputs = feature_extractor(images=img, return_tensors="pt")

        with torch.no_grad():
            outputs = model_vit(**inputs)
            logits = outputs.logits

        predicted_class_index = logits.argmax(-1).item()

        # Get the predicted class label
        predicted_class_label = class_names[predicted_class_index]
        predicted_class_name = class_names[predicted_class_index]

        # Print the prediction
        st.write(f"Ce champignon est de l'éspèce {predicted_class_name} ")

        st.subheader('Shap Interpretation')
        st.image(sha)

        st.subheader('Captum Interpretation')
        st.image(cap)


if page == pages[6]:
    st.header('Interprétabilité ViT')

    option_vit = image_select('Choisissez une image de champignon pour connaitre son éspèce :',
                              [(test_path + 'test.png'), (test_path + 'test2.jpg'), (test_path + 'test3.jpg'),
                               (test_path + 'test4.jpg')])
    img_vit = Image.open(option_vit).resize((224, 224)).convert('RGB')
    col1, col2, col3 = st.columns([3, 2, 3])
    col2.image(img_vit, use_column_width=True, caption='Image à prédire')

    def prediction(option_vit):
        if option_vit == (test_path + 'test.png'):
            sha = pred_path +'vit_test1_shap.png'
            cap = pred_path +'vit_test1_captum.png'
        elif option_vit == (test_path + 'test2.jpg'):
            sha = pred_path +'vit_test2_shap.png'
            cap = pred_path +'vit_test2_captum.png'
        elif option_vit == (test_path + 'test3.jpg'):
            sha = pred_path +'vit_test3_shap.png'
            cap = pred_path +'vit_test3_captum.png'
        elif option_vit == (test_path + 'test4.jpg'):
            sha = pred_path +'vit_test4_shap.png'
            cap = pred_path +'vit_test4_captum.png'
        return sha, cap

    #Load model
    model_save_path = os.path.join(model_path, "vit_model")
    feature_extractor = transformers.ViTFeatureExtractor.from_pretrained(model_save_path)
    model_vit = transformers.ViTForImageClassification.from_pretrained(model_save_path)
    sha, cap = prediction(option_vit)


    # Preprocess the image
    inputs = feature_extractor(images=img_vit, return_tensors="pt")

    with torch.no_grad():
        outputs = model_vit(**inputs)
        logits = outputs.logits

    predicted_class_index = logits.argmax(-1).item()

    # Get the predicted class label
    predicted_class_label = class_names[predicted_class_index]
    predicted_class_name = class_names[predicted_class_index]

    # Print the prediction
    st.write(f"Ce champignon est de l'éspèce {predicted_class_name} ")

    st.subheader('Shap Interpretation')
    st.image(sha)

    st.subheader('Captum Interpretation')
    st.image(cap)

if page == pages[7]:
    st.write('Conclusion')