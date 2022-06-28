import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go

model = tf.keras.models.load_model('./model.h5')
unique = ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale',
       'american_staffordshire_terrier', 'appenzeller',
       'australian_terrier', 'basenji', 'basset', 'beagle',
       'bedlington_terrier', 'bernese_mountain_dog',
       'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound',
       'bluetick', 'border_collie', 'border_terrier', 'borzoi',
       'boston_bull', 'bouvier_des_flandres', 'boxer',
       'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff',
       'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua',
       'chow', 'clumber', 'cocker_spaniel', 'collie',
       'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo',
       'doberman', 'english_foxhound', 'english_setter',
       'english_springer', 'entlebucher', 'eskimo_dog',
       'flat-coated_retriever', 'french_bulldog', 'german_shepherd',
       'german_short-haired_pointer', 'giant_schnauzer',
       'golden_retriever', 'gordon_setter', 'great_dane',
       'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael',
       'ibizan_hound', 'irish_setter', 'irish_terrier',
       'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound',
       'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier',
       'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier',
       'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog',
       'mexican_hairless', 'miniature_pinscher', 'miniature_poodle',
       'miniature_schnauzer', 'newfoundland', 'norfolk_terrier',
       'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog',
       'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian',
       'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler',
       'saint_bernard', 'saluki', 'samoyed', 'schipperke',
       'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier',
       'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier',
       'soft-coated_wheaten_terrier', 'staffordshire_bullterrier',
       'standard_poodle', 'standard_schnauzer', 'sussex_spaniel',
       'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier',
       'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel',
       'west_highland_white_terrier', 'whippet',
       'wire-haired_fox_terrier', 'yorkshire_terrier']


st.markdown("""
    # Dog Breed Predictor
    This app predicts breed of dogs
    ***
         """)

st.sidebar.markdown("# Dog Breed Classifier 🐶")

imgfile = st.file_uploader("Upload Image of your Dog!", type = ['png','jpg','jpeg'])


if imgfile != None:
    image = Image.open(imgfile)
    st.image(image)
    img = np.array(image.convert('RGB'))
    
    img = tf.image.resize(img,(224,224))

    result = model.predict(tf.expand_dims(img,axis=0))
    print(unique[result.argmax()] + ' : ' + str(result.max()))
    
    text_prediction = "We predict ....  \"" + " ".join([i.capitalize() for i in unique[result.argmax()].split("_")]) + '" !!'
    text_confidence = 'With confidence : ' + str(round(result.max(),4)*100)[:5] + "%"
    
    st.markdown(""" <style> .font {
        font-size:30px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.markdown(""" <style> .font2 {
        font-size:25px ; font-family: 'Cooper Black'; color: rgb(54, 173, 84);} 
        </style> """, unsafe_allow_html=True)
    
    st.markdown('<p class="font">{}</p>'.format(text_prediction), unsafe_allow_html=True)
    st.markdown('<p class="font2">{}</p>'.format(text_confidence), unsafe_allow_html=True)


    
    st.subheader('Confidence Level')
    result = result.flatten()
    result = np.around(result,decimals = 4)
    df = pd.DataFrame({'Breed' : unique,'Confidence' : result})
    
    fig = px.pie(df, values='Confidence', names='Breed')
    fig.update_traces(textposition='inside', textfont_size=14)

    st.write(fig)

    
    





