import streamlit as st
import pandas as pd
import numpy as np
from functions import *

filepath = '/Users/filippovicentini/Desktop/programming_project/AmazonSales/datasets/Amazon Sale Report.csv'
df = df_cleaning_modeling_phase(filepath)
#df['amount'].fillna(0,inplace=True)

st.title(':blue[Classification Models] :orange[for Prediction]')

st.markdown("""
We shall make **ORDER REJECTION CLASSIFICATION PREDICTION** using two models based on:
  - **Logistic Regression Classifier**
  - **Random Forest Classifier**
""")

#DATA PREPROCESSING
#CREDO CONVENGA NON CREARE LE FUNZIONI PER CALCOLARE LE CONFUSION MA SCRIVERE QUI DI SEGUITO TUTTO

st.subheader(':blue[Data] :orange[Preprocessing]', divider = 'orange')

target = "rejected"
y = df[target]     # target vector
X = df.drop(target, axis = "columns")   # feature matrix/dataframe


size_test = st.slider('Select the size of the test', min_value=0.0, max_value=1.0, value=0.2)
X_trainlg, X_testlg, y_trainlg, y_testlg = train_test_split(X, y, test_size = size_test, random_state = 42)

# resample it (training dataset) using Random Over Sampling
ros = RandomOverSampler(random_state = 42)
X_train_overlg,y_train_overlg = ros.fit_resample(X_trainlg,y_trainlg)

st.markdown("""In this dataset we have to address  imbalance in the target class distribution in the training-target vector. So we have to resample the training datasets to obtain balanced target class distribution.""")

show_imbalance = st.toggle('Show target class distribution before resampling')
if show_imbalance:
    plot_class_distr(y_trainlg)

show_balance = st.toggle("Show target class distribution after resampling")
if show_balance:
    plot_class_distr(y_train_overlg)
    st.sidebar.info("For predictions we will use the resample dataset!")


#DATA PROCESS

st.subheader(':blue[Process] :orange[Data]', divider='orange')

choose_model = st.radio('Select the Model',['Logistic Regression','Random Forest'],
                        index=None, horizontal=True)

choose_graph = st.radio('Choose the graph', ['Confusion Matrix Training', 'Confusion Matrix Test','ROC Curve'],
                        index=None,horizontal=True)

if choose_model == 'Logistic Regression':
    if choose_graph == 'Confusion Matrix Training':
        confusion_matrix_logistic(X_train_overlg,y_train_overlg)
    if choose_graph == 'Confusion Matrix Test':
        confusion_matrix_logistic(X_testlg, y_testlg)
    if choose_graph == 'ROC Curve':
        roc_curve_logistic(X_testlg, y_testlg)
if choose_model == 'Random Forest':
    if choose_graph == 'Confusion Matrix Training':
        confusion_matrix_forest2(filepath)
    if choose_graph == 'Confusion Matrix Test':
        confusion_matrix_forest()
    if choose_graph == 'ROC Curve':
        roc_curve_rf()
    
















