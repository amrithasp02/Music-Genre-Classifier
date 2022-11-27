#%%writefile app.py
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('music_image_classify.hdf5')
    return model
with st.spinner('Model is being loaded'):
    model=load_model()

st.write(""" # Music Genre Classification""")
st.subheader("Deep CNN model - Squeezenet trained on Mel-Spectrogram images of songs belonging to the following categories: ")
st.text("blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock")
st.set_option('deprecation.showfileUploaderEncoding', False)

file = st.file_uploader("Upload Mel Spectrogram Image", type=["jpg", "png"])

def import_and_predict(image_data, model):
    
        size = (336,217)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
    
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    class_names = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    st.write(score)
    string = "This song belongs to the genre - "+class_names[np.argmax(predictions)]
    st.success(string)
   