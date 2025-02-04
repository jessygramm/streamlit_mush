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

    st.markdown("""
                    As part of the avr24_cds_mushrooms project, we immerse you in the world of **mushrooms.** :herb: These mysterious forest dwellers are fascinating.
                    Their diversity is astounding, ranging from tiny spores invisible to the naked eye to majestic boletes and amanitas.
                    In the kitchen :ramen: mushrooms are treasures of flavors and textures, adding an irresistible touch to many dishes.
                    Additionally, some mushrooms have medicinal properties recognized for millennia, used in both traditional and modern medicine.
                    
                    
                    **However, mushroom picking requires in-depth knowledge, as some species are toxic, even deadly.
                    This is why mushroom recognition is a valuable skill.**
                    
                    
                    Combining technical expertise and business interpretation, the goal of this project is to develop a mushroom recognition system using computer vision
                    algorithms. For training the models, we primarily used data from the [Mushroom Observer](https://mushroomobserver.org/).
                    A rich and collaborative resource aimed at expanding the community of mycology enthusiasts.
                    The data and images are freely available on the site and are regularly enriched by the community.
                    
                    
                    **Below are the main steps of this adventure :**
                    - Understanding the project and the concepts underneath
                    - Exploring the available data
                    - Collecting the image
                    - Preprocessing them (labelling, segmentation, data augmentation,...)
                    - Modeling and training algorithms
                    - Measurement and interpretability
                    """
                )
    with st.expander('About the team', icon=":material/lightbulb:"):
        st.write('''
                         - Jessica Gerstein
                         - Brice Oulion
                         - Napo Koh-Mama
                         - Gaël Ahihounkpe
                         '''
                 )

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
        st.markdown(
            """
            A Convolutional Neural Network uses convolutional layers to automatically detect patterns such as edges, 
            textures, or shapes in images, progressively learning more complex features in deeper layers. These features 
            are passed through pooling layers to reduce spatial dimensions and then through fully connected layers for final 
            classification.
            
            The training process we used : 
            - Instanciate a MLFlow experiment to track the results
            - Start with 'keras_tuner' to help choose hyperparameters
            - Adjusts hyperparameters
            - Random search
            - Test regulation methods to reduce overfitting
            - Modify dataset to deal with class imbalance
            - Retrain and reajust parameters
            """
        )
    with tab2:
        st.header('EfficientNet')
        st.markdown(
            """
            EfficientNet is a family of convolutional neural networks designed for image classification and other vision tasks. 
            It uses a compound scaling method to efficiently balance network depth, width, and resolution, achieving high performance 
            while reducing computational cost. EfficientNet models are lightweight and scalable, making them suitable for 
            resource-constrained environments while delivering state-of-the-art accuracy in image-related tasks.
            
            The training process we used : 
            - Import the EfficientNetB7 model with 'imagenet' weights
            - Add some layers and softmax at the end for classification
            - Compile the model
            - Train it on our data
            - Fine tuning            
            """
        )
    with tab3 :
        st.header('Vision Transformer')
        st.markdown("""
        The Vision Transformer (ViT) is a deep learning model designed for image classification and other computer vision tasks. 
        It applies the transformer architecture, originally developed for natural language processing, to image data. Images are 
        split into fixed-size patches, which are flattened and embedded before being processed by transformer layers. ViT leverages 
        self-attention mechanisms to capture global relationships between image patches, achieving state-of-the-art performance, 
        especially when trained on large datasets.
        
        The training process we used : 
        - Load the 'google/vit-base-patch16-224-in21k' model from Hugging Face :hugging_face:
        - Train it on our datas
        - Test it
        """)

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
            col2.subheader(f"This mushroom is a :green[**{predicted_class_name}**]")

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
            col2.subheader(f"This mushroom is a :green[**{predicted_class_name}**]")


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

    st.markdown("""
                    This project allowed us to deeply explore the application of deep learning algorithms in the fascinating field of mushrooms.
                    At each stage, we faced complex challenges, from data collection and processing to model interpretation,
                    while emphasizing the importance of methodical approaches in data science.
                    In terms of modeling, we optimized several deep learning architectures. The results obtained show satisfactory performance.
                    Although our results are encouraging, it would be interesting to explore other avenues for improvement and expansion of our
                    mushroom classification system:
                    - Explore optimized variants of ViT such as DeiT (Data-efficient Image Transformers) or CvT (Convolutional vision Transformer),
                    which could potentially outperform the standard ViT.
                    - Explore hybrid architectures with models combining the strengths of CNNs and Transformers, like CoAtNet.
                    - Develop few-shot models capable of learning from very few examples, which would be useful for identifying new mushroom species.
                    This work highlights the importance of combining technical expertise and business interpretation.
                    They provide a solid foundation for future research or industrial applications in fields such as biodiversity or mycology,
                    coded on drone videos for example...
                    """)
    st.markdown("![Alt Text](https://i.pinimg.com/originals/65/d4/a3/65d4a33521f6f15d4b8f3b5cdeaec29d.gif)")