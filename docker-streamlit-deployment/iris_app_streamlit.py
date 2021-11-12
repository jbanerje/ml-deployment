from sklearn.utils import shuffle
from io import BytesIO
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import pickle
import base64

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Predicted_data', index=False)
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="iris_prediction.xlsx">Download csv file</a>'

def predict_batch_data(batch_data):
   
   pred_mapping = {0:'SETOSA', 1:'VIRGINICA', 2:'VERSICOLOR'}
   model = pickle.load(open('iris_model.pkl', 'rb'))

   feature_df  = batch_data.drop(columns=['species'])
   target_df   = batch_data['species']

   y_pred = model.predict(feature_df)
   
   y_pred_list = [pred_mapping[i] for i in y_pred ]
   # predicted_df = feature_df
   batch_data['predicted_species'] = y_pred_list
   return batch_data

def pred_iris_flower(input_arr):
   
   """
   Function to predict the iris species based on dimension
   Args:
       input_arr ([type]): Numpy array containing the sepal_length, sepal_width, petal_length, petal_width

   Returns:
       Predicted value - Setosa, Virginica, Versicolor
   """
   
   pred_mapping = {0:'SETOSA', 1:'VIRGINICA', 2:'VERSICOLOR'}
   model = pickle.load(open('iris_model.pkl', 'rb'))
   
   return pred_mapping[model.predict(input_arr)[0]], (model.predict_proba(input_arr).max() * 100).astype(int).astype(str)

def streamlit_interface():
   """
      Function for Streamlit Interface
   """
   st.markdown('<h2 style="background-color:MediumSeaGreen; text-align:center; font-family:arial;color:white">IRIS SPECIES PREDICTION</h2>', unsafe_allow_html=True)
   
   # Sidebars (Left)
   st.sidebar.header("IRIS SPECIES PREDICTION")

   # Sidebar -  Upload File for Batch Prediction
   st.sidebar.subheader("Get Batch Prediction")
   uploaded_file        = st.sidebar.file_uploader("Upload Your .csv File", type='csv', key=None)
   usr_sidebar_model    = st.sidebar.radio('Choose Your Model', ('Random Forest', 'Decision Tree', 'Logistic Regression'))

   if st.sidebar.button('Submit Batch'):
      if uploaded_file is not None:
         batch_data = pd.read_csv(uploaded_file)
         
         # Perform batch Prediction
         batch_pred_df = predict_batch_data(batch_data)

         # Save prediction
         # batch_pred_df.to_csv('./iris_prediction.csv', index=None)
         st.sidebar.text('Prediction Created Sucessfully!')
         st.sidebar.header("Sample Output")
         st.sidebar.text(shuffle(batch_pred_df.head()))
         st.sidebar.header("Download Complete File")
         st.sidebar.markdown(get_table_download_link(batch_pred_df), unsafe_allow_html=True)
   
   # Main Page (Right)
   img = Image.open('./images/iris.JPG')
   st.image(img, width=700)
   sepal_length    = st.slider('Sepal Length', 4.3, 7.9 , step=0.1)
   sepal_width     = st.slider('Sepal Width',  2.0, 4.4 , step=0.1)
   petal_length    = st.slider('Petal Wength', 1.0, 6.9 , step=0.1)
   petal_width     = st.slider('Petal Width',  0.1, 2.5 , step=0.1)

   usr_model = st.selectbox('Choose Your Model', ('Random Forest', 'Decision Tree', 'Logistic Regression'))

   if st.button('Submit'):
      
      image_map = {  'SETOSA'      :  './images/Setosa.JPG',
                     'VIRGINICA'   :  './images/Virginica.JPG',
                     'VERSICOLOR'  :  './images/Versicolor.JPG'
                  }

      input_arr = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)
      iris_prediction, confidence = pred_iris_flower(input_arr)
      
      st.success(usr_model + ' || Prediction : ' + iris_prediction + ' || Model Confidence : ' + confidence + '%')

      st.image(image_map[iris_prediction], width=200, caption = iris_prediction)

if __name__ == '__main__':
    streamlit_interface()