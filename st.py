import tensorflow as tf
import streamlit as st
from PIL import Image

st.title("Welcome to our website!")
@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('C:\\Users\\MY-PC\\Desktop\\DP\\my_model1.hdf5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Facial beauty and age prediction
         """
         )

file = st.file_uploader("Please upload an file", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    
        cv_im = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)

        internal_image = cv2.resize(cv_im,(100,100))
        internal_image = internal_image.reshape(1,100, 100,3)

        p = model.predict(internal_image)

        labels = ['1-4',
                  '5-11',
                  '12-18',
                  '19-26',
                  '27-34',
                  '35-44',
                  '45-64',
                  '65-110']

        pred_list = {x : float(y) for x,y in zip(labels, p[0])}
        pred_list = dict(sorted(pred_list.items(), reverse=True, key=lambda item: item[1]))
        pred_list
        
        return pred_list

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    print(predictions)
