import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

'''
@st.cache(allow_output_mutation=True)
def load_model():
    model = open('svm.pkl', "rb")
    print('MODEL = ',model)
    return model
with st.spinner('Model is being loaded'):
    model=load_model()'''

#model = open('svm.pkl', "rb")
#print('MODEL = ',model)

with open("svm.pkl", "rb") as f:
    model = pickle.load(f)
print('MODEL = ',model)
    
st.write(""" # Music Genre Classification""")
st.subheader("SVM model - trained on Music Features dataset CSV files of songs belonging to the following categories: ")
st.text("blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock")
st.set_option('deprecation.showfileUploaderEncoding', False)

uploaded_file = st.file_uploader("Choose a CSV file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = uploaded_file.getvalue().decode('utf-8').splitlines() 
    del data[0]
    X = np.array([data])
    print(X)
    pred = model.predict(X)
    class_names = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    string = "This song belongs to the genre - "+class_names[pred[0]]
    st.success(string)