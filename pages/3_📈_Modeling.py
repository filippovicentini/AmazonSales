import streamlit as st
import pandas as pd
import numpy as np
from functions import *

filepath = '/Users/filippovicentini/Desktop/programming_project/AmazonSales/datasets/Amazon Sale Report.csv'
df = df_cleaning_modeling_phase(filepath)

st.title(':blue[Classification Models] :orange[for Prediction]')

st.markdown("""
We shall make **ORDER REJECTION CLASSIFICATION PREDICTION** using two models based on:
  - **Logistic Regression Classifier**
  - **Random Forest Classifier**
""")

#DATA PREPROCESSING

st.subheader('Data Preprocessing', divider = 'orange')

target = "rejected"
y = df[target]     # target vector
X = df.drop(target, axis = "columns")   # feature matrix/dataframe
size_test = st.slider('Select the size of the test', min_value=0.0, max_value=1.0, value=0.2)
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size= size_test, random_state = 42)
# resample it (training dataset) using Random Over Sampling
ros = RandomOverSampler(random_state = 42)
X_train_over,y_train_over = ros.fit_resample(X_train,y_train)

st.markdown("""In this dataset we have to address  imbalance in the target class distribution in the training-target vector. So we have to resample the training datasets to obtain balanced target class distribution.""")

show_imbalance = st.toggle('Show target class distribution before resampling')
if show_imbalance:
    plot_class_distr(y_train)

show_balance = st.toggle("Show target class distribution after resampling")
if show_balance:
    plot_class_distr(y_train_over)
    st.sidebar.info("For predictions we will use the resample dataset!")

#DATA PROCESS

st.subheader('Process Data', divider='orange')

choose_model = st.radio('Select the Model',['Logistic Regression','Random Forest'],
                        index=None, horizontal=True)

choose_graph = st.radio('Choose the graph', ['Confusion Matrix Training', 'Confusion Matrix Test','ROC Curve'],
                        index=None,horizontal=True)

if choose_model == 'Logistic Regression':
    if choose_graph == 'Confusion Matrix Training':
        confusion_matrix_logistic(X_train_over,y_train_over)
    if choose_graph == 'Confusion Matrix Test':
        confusion_matrix_logistic(X_test, y_test)















