#%%writefile app.py
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

st.write(""" # Music Genre Classification""")
st.header("UE20CS302 - Machine Intelligence Project")
st.header("Team Members - ")
st.caption(" # Aditya Rao PES1UG20CS022")
st.caption(" # Amritha S Pallavoor PES1UG20CS037")
st.caption(" # Ananya Jalan PES1UG20CS042")
