import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
 
model = load_model(r"C:\Users\ASUS\plant_disease\plant_disease.h5")
 
 
CLASS_NAMES = ['Corn-Common_rust','Potato-Early_blight','Tomato-Bacteria_spot']
 
col1,col2 = st.columns([1,2])
 
with col2:
    
     st.title('Plant Disease Detection')
     st.markdown("Upload an Image of the Plant Leaf")

     plant_image = st.file_uploader("Choose an Image....",type = "jpg")
     submit = st.button('Predict')
     
      
if submit:
    
    
    if plant_image is not None:
        file_bytes = np.asarray(bytearray(plant_image.read()),dtype = np.uint8)
        opencv_image = cv2.imdecode(file_bytes,1)
        
        
        
        st.image(opencv_image,channels = 'BGR')
        st.write(opencv_image.shape)
        
        opencv_image = cv2.resize(opencv_image,(256,256))
        
        opencv_image.shape=(1,256,256,3)
        
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        
        st.title(str("This is "  +result.split('-')[0]+"leaf with" + result.split('-')[1]))