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

st.title(":mushroom: Mushroom Prediction :mushroom:")

st.sidebar.image('data/mush.png', width=200)

pages=["Introduction","Data Gathering","Exploration", "Pre Processing", "Modelisation", "Predict & Interpret","Conclusion"]
page=st.sidebar.radio("Go to",pages)

if page == pages[0]:
    st.write('Introduction')

if page == pages[1]:
    st.write('Data Gathering')

if page == pages[2]:
    st.write('Exploration')

if page == pages[3]:
    st.write('Pre Processing')

if page == pages[4]:
    st.write('We created and trained several models to achieve the most reliable results possible. '
             'For some, we built them from scratch, while for others, we used transfer learning techniques. ')
    st.write('Here is a summary of the results obtained.:')
    results = pd.DataFrame({
        "Model": ["CNN", "MobileNet", "EfficientNetB3", "YoloV8l", "VIT"],
        "F1 score train": [0.98, 0.98, 0.99, 0.94, 1.0],
        "F1 score test": [0.89, 0.91, 0.93, 0.90, 0.95]
    })
    results.set_index("Model", inplace = True)
    results = results.sort_values('F1 score test', ascending=True)
    st.dataframe(results.style.highlight_max(axis=0), use_container_width=True)

    st.write('Below you can find details on three models used in this project')
    st.write(':bulb: Note that you can test EfficientNet and Transformer on the next page!')

    tab1, tab2, tab3 = st.tabs(['CNN', 'EfficientNet', 'Vision Transformer'])
    with tab1:
        st.header('ConvNet')
        st.write('What we did ...')
    with tab2:
        st.header('EfficientNet')
        st.write('What we did ...')
    with tab3 :
        st.header('Vision Transformer')
        st.write('What we did ...')

if page == pages[5]:

    option = image_select("Choose a mushroom image :", [(test_path + 'test1.jpg'), (test_path + 'test2.jpg'), (test_path + 'test3.jpg'), (test_path + 'test4.jpg')])
    img = Image.open(option).resize((224, 224)).convert('RGB')
    col1, col2, col3 = st.columns([3,2,3])
    col2.image(img, use_column_width=True, caption='Image to predict')

    left, right = st.columns(2)
    if left.button("EfficientNet Model", use_container_width=True):
        def prediction(option):
            if option == (test_path + 'test1.jpg'):
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
        with st.container(border = True):
            col1, col2, col3 = st.columns([1, 4, 0.1])
            col2.subheader(f"This mushroom is a :green-background[**{predicted_class_name}**]")

        st.subheader('Gradcam Interpretation', divider = 'gray')
        st.image(grad)
        with st.expander('Gradcam Explanation', icon = ":material/lightbulb:"):
            st.write('''
            Gradcam is a technique for visualizing the regions of an input image that contribute most to a deep learning model's 
            prediction. It uses the gradients of the target class with respect to the final convolutional layer’s feature maps 
            to generate a weighted heatmap, highlighting important areas of the image. Grad-CAM produces a coarse localization 
            map that shows which regions are most influential for the classification decision. This method is especially useful 
            for convolutional neural networks, providing insights into model behavior by revealing spatial patterns associated 
            with predictions.
            ''')

        st.subheader('Shap Interpretation', divider = 'gray')
        col1, col2, col3 = st.columns([0.1, 4, 0.1])
        col2.image(sh, use_column_width=True)
        with st.expander('Shap Explanation', icon = ":material/lightbulb:"):
            st.write('''
            Shap applies game theory to break down complex model predictions into contributions from each pixel. It
            calculates Shapley values, which represent the average contribution of each feature to the prediction, 
            considering all possible feature subsets. This is done by approximating the prediction for every possible 
            combination of features, enabling the explanation of how individual pixels or regions in an image influence the 
            output. SHAP provides both local and global interpretability.
            ''')

    if right.button("Transformer Model :hugging_face: ", use_container_width=True):
        def prediction(option):
            if option == (test_path + 'test1.jpg'):
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
        with st.container(border=True):
            col1,col2,col3 = st.columns([1, 4, 0.1])
            col2.subheader(f"This mushroom is a :green-background[**{predicted_class_name}**]")


        st.subheader('Shap Interpretation', divider = 'gray')
        st.image(sha)
        with st.expander('Shap Explanation', icon = ":material/lightbulb:"):
            st.write('''
            Shap applies game theory to break down complex model predictions into contributions from each pixel. It
            calculates Shapley values, which represent the average contribution of each feature to the prediction, 
            considering all possible feature subsets. This is done by approximating the prediction for every possible 
            combination of features, enabling the explanation of how individual pixels or regions in an image influence the 
            output. SHAP provides both local and global interpretability.
            ''')

        st.subheader('Captum Interpretation', divider = 'gray')
        st.image(cap)
        with st.expander('Captum Explanation', icon = ":material/lightbulb:"):
            st.write('''
            Captum is a model interpretability library that provides a suite of attribution algorithms for understanding and 
            visualizing model predictions. It includes methods like Integrated Gradients, Layer Conductance, and Saliency, 
            which attribute the importance of each input pixel to the model’s output. Captum works by computing gradients or 
            other relevance scores to highlight which parts of the input were most influential.
            ''')

if page == pages[6]:
    st.write('Conclusion')