'''To be design using streamlit'''
import streamlit as st
import numpy as np
import pandas as pd
import xlwings as xw 
from src.components import data_ingestion
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
import pdb

st.set_page_config(
    page_title="Ashad ML projects",
    page_icon=":rocket:",
    layout="wide"
    )

def home():
    
    st.write("Welcome to my ML prediction Model 👋")
    
    # st.sidebar.success("TradeHull Home")

    # st.markdown(
    #     """
    #     ### This is Trade Hull Alice Blue Copy Trade Software

    #     **👈 NAVIGATE to different pages using Dropdown**
        

    # """
    #     )

def user_inputs():
    
    st.write("# Fill The Required Features For Prediction")
    st.sidebar.success("SCORE PREDICTION")

    df = get_data()
    gender_option = list(df['gender'].unique())
    race_option = list(df['race/ethnicity'].unique())
    parent_option = list(df['parental level of education'].unique())
    lunch_option = list(df['lunch'].unique())
    test_course_option = list(df['test preparation course'].unique())
    
    
    with st.form(key='1',clear_on_submit=True):
        col1,col2,col3= st.columns([5,5,5])
        with col1:
            Gender=st.selectbox('**Gender**',gender_option,key=1)
        with col2:
            Race_Ethnicity=st.selectbox('**Race Ethnicity**',race_option,key=2)
        with col3:
            parental_level_of_education=st.selectbox('**Parents Education**',parent_option,key=3)   
        Lunch=st.selectbox('**Lunch**',lunch_option,key=4)
        test_preparation_course=st.selectbox('**Test Preparation Course**',test_course_option,key=5)
        reading_score=st.slider('**Reading Score**',min_value=0,max_value=100)
        writing_score=st.slider('**Writting Score**',min_value=0,max_value=100)
        submit_button=st.form_submit_button(label='PREDICT MATH SCORE')

        data = CustomData(Gender,Race_Ethnicity,parental_level_of_education,Lunch,test_preparation_course,reading_score,writing_score)
        features = data.get_data_as_data_frame()
        
        if submit_button:
            predict_pipeline = PredictPipeline()
            preds = predict_pipeline.predict(features)
            st.success(f'The Predicted Math Score is {round(preds[0],1)}')

def get_data():
    obj = data_ingestion.Dataingestion()
    train_data,test_data,raw_data = obj.initiate_data_ingestion()
    df = pd.read_csv(raw_data)
    return df
    # st.checkbox("Use container width", value=False, key="use_container_width")
    # st.dataframe(df,use_container_width=st.session_state.use_container_width)

page_names_to_funcs = {
    
    "Home": home,
    "Features": user_inputs,
    "Data":get_data,
    # "Add Clients":Feed_clients
    
}

demo_name = st.sidebar.selectbox("Choose Pages", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()

# streamlit run application.py
# if __name__ == "__main__":
#     get_data()
