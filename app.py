
import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import  mean_squared_error 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error

from sklearn.neighbors import KNeighborsRegressor
from PIL import Image

#app=Flask(__name__)
#Swagger(app)
import os
os.chdir(r'C:\Users\Yogesh\python learn\Praxis\DMD')
pickle_in = open('pickle_file_car.pkl',"rb")
classifier=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def Flower_dectator(sepal_length,sepal_width,petal_length,petal_width):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: sepal_length
        in: query
        type: number
        required: true
      - name: sepal_width
        in: query
        type: number
        required: true
      - name: petal_length
        in: query
        type: number
        required: true
      - name: petal_width
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
   
    prediction=classifier.predict([[sepal_length,sepal_width,petal_length,petal_width]])
    print(prediction)
    return prediction



def main():
    st.title("Car MPG Predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;"> Car MPG Predictor Web App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    sepal_length = st.text_input("Cylinders","Type Here number of Cylinders (8, 4, 6, 3, 5)")
    sepal_width = st.text_input("Horsepower","Type Here Horsepower range between (0-250)")
    petal_length = st.text_input("Weight","Type Here Weight range between (600-6000)")
    petal_width = st.text_input("Acceleration","Type Here Horsepower range between (8-25)")
    result=""
    if st.button("Predict"):
        result=Flower_dectator((sepal_length),(sepal_width),(petal_length),(petal_width))
        result = np.round(result)[0]
        result = result[0] 
    st.success('The predicted Car MPG is {}'.format(result))
    if st.button("About"):
        st.text("This apps help to predict Car's MPG")
        st.text("Built with Streamlit")

os.chdir(r'C:\Users\Yogesh\python learn\krish')

if __name__=='__main__':
    main()


    
    