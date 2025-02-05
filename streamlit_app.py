import os
import transformers
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import time
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
dg_path = 'img_datagathering/'
prep_path = 'img_preprocessing/'

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

# load and cache dataset
@st.cache_data
def load_dataset():
    return pd.read_csv('data/df_16000p_end.csv')

# load df
df = load_dataset()

st.title(":mushroom: Mushroom Prediction :mushroom:")


st.sidebar.image('data/mush.png', width=200)

pages=["Introduction","Data Gathering","Exploration", "Pre Processing", "Modelisation", "Predict & Interpret","Conclusion"]
page=st.sidebar.radio("Go to",pages)

if page == pages[0]:
    st.write("   ")
    st.write("   ")

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

    imageMOL = Image.open(dg_path + "MObsv_logo.png")
    imagepbs = Image.open(dg_path + "pybs4.png")
    imageINT = Image.open(dg_path + "international.jpeg")
    imagePAR = Image.open(dg_path + "participatif.jpeg")
    imagearr = Image.open(dg_path + "arrow.png")
    image0 = Image.open(dg_path + "website.jpg")
    image1 = Image.open(dg_path + "website1.png")
    image2 = Image.open(dg_path + "website2.png")
    image3 = Image.open(dg_path + "website3.png")
    image4 = Image.open(dg_path + "website4.png")
    flvl1 = Image.open(dg_path + "flvl1.png")
    flvl2 = Image.open(dg_path + "flvl2.png")
    flvl3 = Image.open(dg_path + "flvl3.png")
    flvl4 = Image.open(dg_path + "flvl4.png")
    flvl5 = Image.open(dg_path + "flvl5.png")
    slvl = Image.open(dg_path + "slvl.png")
    slvl1 = Image.open(dg_path + "slvl1.png")
    slvl2 = Image.open(dg_path + "slvl2.png")
    slvl3 = Image.open(dg_path + "slvl3.png")

    st.write("   ")
    st.write("   ")    
    st.write("   ")

    st.markdown("""<h1 style='text-align: center; text-decoration: underline;'>Data Gathering</h1>""", unsafe_allow_html=True)

    st.markdown("""<h2 style='text-align: center;'>Where Did We Get the Data?</h2>""", unsafe_allow_html=True)

    st.markdown("""<div style="text-align: justify;">
        The first step was to collect a large dataset on mushrooms to train our deeplearning 
        models effectively. We needed images with correctly identified species names.
        We decided to use MushroomObserver.org, a collaborative American platform where users 
        worldwide contribute to mushroom identification.<div>
        """,unsafe_allow_html=True)

    st.write("   ")
    st.write("   ")
    st.write("   ")    

    with st.expander("Website", expanded=True):

        st.markdown("<h4 style='text-align: center;'> More Details on MushroomObserver.org</h4>", unsafe_allow_html=True)

        st.write("   ")

        W, col1, X, col2, Y, col3, Z = st.columns([1,2,1,2,1,2,1])
        
        col2.image(imageINT)
        col1.image(imageMOL)
        col3.image(imagePAR)

        st.write("   ")

        X, col1, col2, col3, col4, col5 = st.columns([1,3,3,3,3,4])

        col1.metric("Countries worldwide", "152")
        col2.metric("Identified specie", "1,873")
        col3.metric("Active contributors", "+ 13,000")
        col4.metric("Observations", "≃ 530,000")
        col5.metric("Images", "≃ 1,725,000")

    st.write("   ")
    st.markdown("""<h2 style='text-align: center;'>What is the structure of the website?</h2>""", unsafe_allow_html=True)

    if "animation1_start" not in st.session_state:
        st.session_state.animation1_start = False
    if "view1_img" not in st.session_state:
        st.session_state.view1_img = False

    with st.expander("Structure", expanded=True):

        colW, col1, colX, col2, colY, col3, colZ = st.columns([1,1,1,1,1,1,1])

        if col1.button("Start"):
            st.session_state.view1_img = False
            st.session_state.animation1_start = True
        if col2.button("Freeze"):
            st.session_state.animation1_start = False
            st.session_state.view1_img = True
        if col3.button("Hide"):
            st.session_state.animation1_start = False
            st.session_state.view1_img = False

        st.write("   ") 

        X, col1, Y, col2, Z = st.columns([1,9,1,4,1])
        image_container1 = col1.empty()

        if st.session_state.animation1_start:
            col2.metric("Mushrooms / page", "12")
            col2.metric("Total Pages", "44,000")
            col2.markdown("""<div style="text-align: justify;"> 
                Instead of retrieving the entire site, which would have been too time-consuming, we 
                prioritized the most reliable identifications, sorting pages by their confidence degree.<div>
                """, unsafe_allow_html=True)

            for _ in range(1000):
                image_container1.image(image1, use_column_width=True)
                time.sleep(1)
                image_container1.image(image2, use_column_width=True)
                time.sleep(1)
                image_container1.image(image3, use_column_width=True)
                time.sleep(0.5)
                image_container1.image(image4, use_column_width=True)
                time.sleep(1)

                if not st.session_state.animation1_start:
                    break

            image_container1.empty()
            col2.empty()

        elif st.session_state.view1_img:
            col2.metric("Mushrooms / page", "12")
            col2.metric("Total Pages", "44,000")
            col2.markdown("""<div style="text-align: justify;"> 
                Instead of retrieving the entire site, which would have been too time-consuming, we 
                prioritized the most reliable identifications, sorting pages by their confidence scores.<div>
                """, unsafe_allow_html=True)

            image_container1.image(image0, use_column_width=True)

        elif not st.session_state.animation1_start and not st.session_state.view1_img:
            image_container1.empty()
            col2.empty()

    st.write("   ")
    st.markdown("""<h2 style='text-align: center;'>What do we need?</h2>""", unsafe_allow_html=True)



    if "animation2_start" not in st.session_state:
        st.session_state.animation2_start = False
    if "animation3_start" not in st.session_state:
        st.session_state.animation3_start = False
    if "view2_img" not in st.session_state:
        st.session_state.view2_img = False

    with st.expander("Data of interest", expanded=True):

        st.markdown("<h4 style='text-align: center;'>Some data hide in 2 levels</h4>", unsafe_allow_html=True)

        V, col1, W, col2, X, col3, Y, col4, Z = st.columns([1,1,1,1,1,1,1,1,1])

        if col1.button("1st level"):
            st.session_state.animation2_start = True
            st.session_state.animation3_start = False
            st.session_state.view2_img = False
        if col2.button("2nd level"):
            st.session_state.animation3_start = True
            st.session_state.animation2_start = False
            st.session_state.view2_img = False
        if col3.button("Overview"):
            st.session_state.animation2_start = False
            st.session_state.animation3_start = False
            st.session_state.view2_img = True
        if col4.button("Hide "):
            st.session_state.animation2_start = False
            st.session_state.animation3_start = False
            st.session_state.view2_img = False

        if st.session_state.animation2_start:

            Y, col5, Z = st.columns([1,5,1,])
            image_container = col5.empty()

            for _ in range(1000):
                image_container.image(flvl1, use_column_width=True)
                time.sleep(0.3)
                image_container.image(flvl2, use_column_width=True)
                time.sleep(0.3)
                image_container.image(flvl3, use_column_width=True)
                time.sleep(0.3)
                image_container.image(flvl4, use_column_width=True)
                time.sleep(0.3)
                image_container.image(flvl5, use_column_width=True)
                time.sleep(3)
                if not st.session_state.animation2_start:
                    break
            image_container.empty()

        elif st.session_state.animation3_start:

            Y, col6, Z = st.columns([1,18,1,])
            image_container = col6.empty()

            for _ in range(1000):
                image_container.image(slvl, use_column_width=True)
                time.sleep(0.3)
                image_container.image(slvl1, use_column_width=True)
                time.sleep(0.3)
                image_container.image(slvl2, use_column_width=True)
                time.sleep(0.3)
                image_container.image(slvl3, use_column_width=True)
                time.sleep(3)
                if not st.session_state.animation3_start:
                    break
            image_container.empty()

        elif st.session_state.view2_img:
            col7, Z, col8 = st.columns([33.3,1,44])
            with col7:
                st.image(flvl5, use_column_width=True)
            with col8:
                st.image(slvl3, use_column_width=True)

    col1, col2= st.columns(2)
    
    col1.markdown("""<h3 style='text-align: center;'>In the main page : level 1</h3>""", unsafe_allow_html=True)
    col2.markdown("""<h3 style='text-align: center;'>In observation page : level 2</h3>""", unsafe_allow_html=True)

    X, col1, Y, col2, Z = st.columns(5)
    col1.markdown("""
    - observation ID 
    - species name
    - date
    - location""")
    col2.markdown("""
    - images IDs
    - precise location
    - confidence degree""")

    col2.markdown("""<div style='text-align: center;'>(🔒 Confidence degrees were only 
    accessible after logging into a user account)<div>""", unsafe_allow_html=True)

    st.write("   ")
    st.markdown("""<h2 style='text-align: center;'> How did we get it?</h2>""", unsafe_allow_html=True)
    st.write("   ")
    st.write("   ")

    U, col1, V, col2, W, col3, X, col4, Y, col5, Z = st.columns([0.4,3,0.6,1,0.4,2,0.1,1,0.4,5,0.4])

    col2.write("   ")
    col2.write("   ")
    col3.write("   ")
    col4.write("   ")
    col4.write("   ")
    col5.write("   ")
    col2.write("   ")
    col3.write("   ")
    col4.write("   ")
    col5.write("   ")
    col3.write("   ")
    col5.write("   ")

    col1.image(imagepbs, use_column_width=True)
    col2.image(imagearr, use_column_width=True)
    col3.markdown("""<h3 style='text-align: center; text-decoration: bold;'>RUN</h3>""", unsafe_allow_html=True)
    col4.image(imagearr, use_column_width=True)
    col5.markdown("""<h3 style='text-align: center; text-decoration: underline;'>mushroom_raw_data.csv</h3>""", unsafe_allow_html=True)

    st.write("   ")
    st.markdown("""<h3 style='text-align: center;'>Ethical considerations</h3>""", unsafe_allow_html=True)

    st.write("<div style='text-align: center;'>To ensure ethical and responsible scraping, we added delays between"
        "requests to avoid overloading the site and interfering with other users.<div>", unsafe_allow_html=True)
    st.write("<div style='text-align: center;'>We also implemented 10-minute breaks every hour to prevent bans and reduce server strain.<div>", unsafe_allow_html=True)
    st.markdown("""<h3 style='text-align: center;'> </h3>""", unsafe_allow_html=True)

    st.markdown("""<h2 style='text-align: center;'>Short overview of the data collected</h2>""", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Pages Scraped", "16,000")
    col2.metric("Total Identifications", "192,000")
    col3.metric("Confidence Score Range", "98% - 70%")
    col4.metric("Id of potential images","588,000")

    st.markdown("""<h3 style='text-align: center;'> </h3>""", unsafe_allow_html=True)

    st.markdown("""<h2 style='text-align: center;'>Extracted Mushroom Data</h2>""", unsafe_allow_html=True)
    st.write("   ")

    mushroom_data = pd.DataFrame({
        "Image IDs": [
            "183600;183601", "183604;183605", "183606;183607", "183610;183611;183612;183613", "183615;183614"
        ],
        "Title": [
            "Leccinum manzanitae", "Chroogomphus ochraceus", "Boletus eastwoodiae", "Armillaria", "Morganella pyriformis"
        ],
        "Location": [
            "USA, California", "USA, California", "USA, California", "USA, California", "USA, California"
        ],
        "Date": [
            "2011-11-20", "2011-11-20", "2011-11-20", "2011-11-20", "2011-11-20"
        ],
        "Latitude": [
            37.83294, 37.83294, 37.83294, 37.83294, 37.83294
        ],
        "Longitude": [
            -122.16564, -122.16564, -122.16564, -122.16564, -122.16564
        ],
        "Confidence Score": [
            "85%", "85%", "85%", "85%", "80%"
        ]
    }, index=[82884, 82886, 82887, 82889, 82890])

    st.dataframe(mushroom_data)

if page == pages[2]:
    st.write("   ")
    st.write("   ")
    st.write('We started by exploring the dataset to understand its structure and characteristics. '
             'We examined the distribution of features, and the characteristics of the images. '
             'This initial exploration helped us identify areas for improvement and guide our data preprocessing efforts.')


    # Cache the data processing functions
    @st.cache_data
    def process_species_data(df):
        species_df = df[df['species'].str.split().str.len() > 1]
        species_with_images = species_df[species_df['image_ids'].notna()]
        return species_with_images


    @st.cache_data
    def get_country_data():
        df = pd.read_csv('data/filtered_mushroom_dataset.csv')
        df['country'] = df['location'].str.split(',').str[0]
        df = df[df['species'] != 'Mixed collection']
        return df


    @st.cache_data
    def create_species_plot(species_with_images):
        top_species = species_with_images['species'].value_counts().head(50)
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(top_species.index, top_species.values,
                      color=plt.cm.viridis(top_species.values / max(top_species.values)))
        plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax, label='Count')
        ax.set_title('top 50 Species with Images')
        ax.set_xlabel('Species')
        ax.set_ylabel('Count')
        plt.xticks(rotation=90)
        plt.tight_layout()
        return fig


    # Process data once
    species_with_images = process_species_data(df)

    # Display df 10 first rows
    if st.checkbox('Display df sample'):
        st.write('Here is a sample of the dataset:')
        st.dataframe(df.head(10))

        # df shape
        st.write('The dataset has', species_with_images.shape[0], 'rows and', species_with_images.shape[1], 'columns.')

    # display top 50 species with images
    if st.checkbox('Display top 50 species distribution'):
        fig = create_species_plot(species_with_images)
        st.pyplot(fig)

    # display distribution per country
    if st.checkbox('Display distribution per country'):
        country_df = get_country_data()

        # Calculate countries per species
        countries_per_species = (country_df.groupby('species')['country']
                                 .nunique()
                                 .reset_index(name='num_countries')
                                 .sort_values('num_countries', ascending=False))

        st.write("Number of countries per species:")
        st.dataframe(countries_per_species)

        # Create country distribution plot
        fig, ax = plt.subplots(figsize=(12, 8))
        country_counts = country_df['country'].value_counts()
        bars = ax.bar(country_counts.index, country_counts.values,
                      color=plt.cm.viridis(country_counts.values / max(country_counts.values)))
        plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax, label='Count')
        ax.set_title('Distribution per country')
        ax.set_xlabel('Country')
        ax.set_ylabel('Count')
        plt.xticks(rotation=90)
        plt.tight_layout()
        st.pyplot(fig)

        st.image('data/countryMap.png', use_column_width=True)


    # Cache the species images display
    @st.cache_data
    def display_species_grid():
        species_names = list(class_names.values())  # Convert the class_names dictionary values to a list
        cols = st.columns(5)
        for idx, species in enumerate(species_names[:13]):
            col_idx = idx % 5
            with cols[col_idx]:
                st.write(f"**{class_names[idx]}**")
                species_count = df[df['species'] == class_names[idx]].shape[0]
                st.write(f"*({species_count} images)*")
                st.image(f'data/species/{species}.png', use_column_width=True)


    # display images of species
    if st.checkbox('Display species images'):
        display_species_grid()

if page == pages[3]:
    imagearr = Image.open(prep_path + "arrow.png")
    imagebox = Image.open(prep_path + "box.png")
    imageboxt = Image.open(prep_path + "box_train.png")
    imagerb1 = Image.open(prep_path + "resbox2.png")
    imagerb2 = Image.open(prep_path + "resbox1.png")
    imagecrop = Image.open(prep_path + "tocrop.jpg")
    imagecrop1 = Image.open(prep_path + "crop1.png")
    imagecrop2 = Image.open(prep_path + "crop2.png")
    imagecrop3 = Image.open(prep_path + "crop3.png")
    imagedsetb = Image.open(prep_path + "dsetbase.png")
    imagedsete = Image.open(prep_path + "dsetequi.png")
    imagers1 = Image.open(prep_path + "resseg1.png")
    imagers2 = Image.open(prep_path + "resseg2.png")
    imageseg = Image.open(prep_path + "seg.png")
    imagesegt = Image.open(prep_path + "seg_train.png")

    st.markdown("""<h1 style='text-align: center; text-decoration: underline;'>Pre-Processing</h1>""", unsafe_allow_html=True)

    st.markdown("""<h2 style='text-align: center;'>Preparing Images for Models</h2>""", unsafe_allow_html=True)

    st.markdown("""<div style='text-align: center;'>In this stage of the project, the goal was to preprocess image 
        data efficiently to train our detection and classification models.</div>""", unsafe_allow_html=True) 
    st.markdown("""<div style='text-align: center;'>We followed several key steps to enhance dataset quality and 
        optimize model performance.</div>""", unsafe_allow_html=True)

    st.markdown("""<h2 style='text-align: center;'>Bounding Boxes</h2>""", unsafe_allow_html=True)
    
    st.write("   ")
    st.markdown("""<div style='text-align: center;'> 
        We chose YOLO models for its speed and accuracy and because it is esay to work with it.</div>""", 
        unsafe_allow_html=True)
    st.write("   ")
    st.write("   ")

    col1, Y, col2, Z, col3 = st.columns([3,0.2,1,0.2,6])

    col2.write("   ")
    col2.write("   ")
    col2.write("   ")
    col2.write("   ")
    col2.write("   ")
    col2.write("   ")
    col2.write("   ")
    col2.write("   ")
    col2.write("   ")

    col2.image(imagearr, caption="Training Yolo models", use_column_width=True)
    col1.image(imagebox, caption="boxing with LabelImg", use_column_width=True)
    col3.image(imageboxt, caption="Prediction of Yolo", use_column_width=True)

    st.markdown("""<div style='text-align: center;'> 
        We selected 120 images per species  ➡️  manually annotated (LabelImg)  ➡️  training YOLO 
        models</div>""", unsafe_allow_html=True)
    st.write("   ")
    st.write("   ")
    
    with st.expander("YoloV8l Performance"):
        col1, col2 = st.columns([1,1])

        st.image(imagerb1, use_column_width=True)
        st.image(imagerb2, use_column_width=True)
 
    st.markdown("""<h2 style='text-align: center;'>Image cropping</h2>""", unsafe_allow_html=True)

    st.markdown("""<div style='text-align: center;'>We used the label obtained during the boxing stage
    to crop the images and keep only the interesting part and reduce the image size.</div>""", unsafe_allow_html=True)
    st.write("   ")
    st.write("   ")

    col1, Y, col2, Z, col3 = st.columns([30,1,5,1,6])

    col2.write("   ")
    col2.write("   ")
    col2.write("   ")
    col1.image(imagecrop, caption="Initial image",use_column_width=True) 
    col2.image(imagearr, caption=" ", use_column_width=True)
    col2.write("   ")
    col2.write("   ")
    col2.write("   ")
    col2.write("   ")
    col2.image(imagearr, caption=" ", use_column_width=True)
    col2.write("   ")
    col2.write("   ")
    col2.write("   ")
    col2.write("   ")
    col2.write("   ")
    col2.write("   ")
    col2.image(imagearr, use_column_width=True)
    col3.image(imagecrop1, use_column_width=True)
    col3.write("   ")
    col3.image(imagecrop2, use_column_width=True)
    col3.write("   ")
    col3.image(imagecrop3, caption="3 croped images", use_column_width=True)
    
    st.markdown("""<h2 style='text-align: center;'>Ballenced data set</h2>""", unsafe_allow_html=True)

    st.markdown("""<div style='text-align: center;'>A well-balanced dataset prevents the model from favoring
        overrepresented species.</div>""", unsafe_allow_html=True)
    st.write("   ")
    st.write("   ")

    col1, Y, col2, Z, col3 = st.columns([10,0.4,2,0.4,9])

    col2.write("   ")
    col2.write("   ")
    col2.write("   ")
    col2.write("   ")
    col2.write("   ")
    col2.write("   ")

    col1.image(imagedsetb, caption="Initial dataset distribution", use_column_width=True)
    col2.image(imagearr, caption=" ", use_column_width=True)
    col3.image(imagedsete, caption="Balanced dataset distribution", use_column_width=True)

    st.markdown("""<div style='text-align: center;'>Some species were overrepresented, while others had fewer samples. 
        </div>""", unsafe_allow_html=True)

    st.markdown("""<h2 style='text-align: center;'>Data augmentation</h2>""", unsafe_allow_html=True)

    st.markdown(
        """
        To enhance dataset diversity, we applied various **data augmentation techniques**, including:
        - **Geometric transformations** (rotations, flips, zooms...)
        - **Photometric modifications** (contrast, brightness...)
        - **SMOTE (Synthetic Minority Over-sampling Technique)** for generating additional samples in minority classes.
        
        However, excessive augmentation created **storage and training constraints** that required more optimization.
        """
    )

    st.markdown("""<h2 style='text-align: center;'>Segmentation</h2>""", unsafe_allow_html=True)
    st.markdown("""<div style='text-align: center;'>We performed mushroom segmentation to precisely 
    isolate their shapes and try to apply masks.</div>""", unsafe_allow_html=True)

    st.write("   ")
    st.write("   ")

    col1, Y, col2, Z, col3 = st.columns([5,0.2,1,0.2,6])

    col2.write("   ")
    col2.write("   ")
    col2.write("   ")
    col2.write("   ")
    col2.write("   ")
    col2.write("   ")
    col2.write("   ")
    col2.write("   ")
    col2.write("   ")
    col2.image(imagearr, caption="Training Yolo models", use_column_width=True)
    col1.image(imageseg, caption="boxing with LabelImg", use_column_width=True)
    col3.image(imagesegt, caption="Prediction of Yolo", use_column_width=True)

    st.markdown("""<div style='text-align: center;'> 
        We selected 120 images per species  ➡️  manually annotated (Labelme)  ➡️  training YOLO 
        models</div>""", unsafe_allow_html=True)
    
    st.write("   ")
    st.write("   ")

    with st.expander("YoloV8l Performance"):

        col1, col2 = st.columns([1,1])

        st.image(imagers1, use_column_width=True)
        st.image(imagers2, use_column_width=True)

if page == pages[4]:
    st.write("   ")
    st.write("   ")
    st.write('We created and trained several models to achieve the most reliable results possible. '
             'For some, we built them from scratch, while for others, we used transfer learning techniques. ')
    st.write('Here is a summary of the results :')
    results = pd.DataFrame({
        "Model": ["CNN", "MobileNet", "EfficientNetB3", "YoloV8l", "VIT"],
        "F1 score train": [0.98, 0.98, 0.99, 0.94, 1.0],
        "F1 score test": [0.89, 0.91, 0.93, 0.90, 0.95]
    })
    results.set_index("Model", inplace = True)
    results = results.sort_values('F1 score test', ascending=True)
    st.dataframe(results.style.highlight_max(axis=0), use_container_width=True)

    st.write('Below you can find details on three models used in this project')
    st.write(':bulb: Note that you can test EfficientNet and Transformer models on the next page!')

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
            Grad-CAM (Gradient-weighted Class Activation Mapping): Generates heatmaps by using the gradients of convolutional 
            layers to highlight which parts of the image influenced the prediction. Specific to CNNs, highly visual, but less 
            precise than SHAP or IG for fine-grained explanations.
            ''')

        st.subheader('Shap Interpretation', divider = 'gray')
        col1, col2, col3 = st.columns([0.1, 4, 0.1])
        col2.image(sh, use_column_width=True)
        with st.expander('Shap Explanation', icon = ":material/lightbulb:"):
            st.write('''
            SHAP (SHapley Additive exPlanations): Based on Shapley value theory, SHAP assigns importance to each pixel by 
            evaluating its impact on the prediction when removed or modified. It provides both global and local explanations 
            but is computationally expensive for deep models.
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
            SHAP (SHapley Additive exPlanations): Based on Shapley value theory, SHAP assigns importance to each pixel by 
            evaluating its impact on the prediction when removed or modified. It provides both global and local explanations 
            but is computationally expensive for deep models.
            ''')

        st.subheader('Integrated Gradients Interpretation', divider = 'gray')
        st.image(cap)
        with st.expander('Integrated Gradients Explanation', icon = ":material/lightbulb:"):
            st.write('''
            Integrated Gradients (IG): A gradient-based method that assigns importance to pixels by integrating the 
            gradients between a reference image (often a black image) and the input image. Faster than SHAP and well-suited 
            for deep neural networks.
            ''')

if page == pages[6]:
    st.write("   ")
    st.write("   ")
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
