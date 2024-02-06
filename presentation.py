import streamlit as st
import pandas as pd
import numpy as np

filepath = '/Users/filippovicentini/Desktop/programming_project/AmazonSales/datasets/Amazon Sale Report.csv'


def section_introduction():
    st.title("Amazon Sale Report")
    st.write("In this web application blablabla...")



def section_data_loading():
    st.title("Sezione 2: Caricamento dei Dati")
    uploaded_file = st.file_uploader("Carica il tuo file CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Ecco le prime righe del tuo dataset:")
        st.write(df.head())

def section_data_analysis():
    st.title("Sezione 3: Analisi dei Dati")
    # Aggiungi il codice per eseguire l'analisi dei dati qui

def section_results_conclusions():
    st.title("Sezione 4: Risultati e Conclusioni")
    # Aggiungi il codice per visualizzare i risultati e le conclusioni qui

# Barra laterale per la navigazione
section = st.sidebar.radio("Navigation", ["Dataset Information", "Caricamento dei Dati", "Analisi dei Dati", "Risultati e Conclusioni"])

# Contenuto principale in base alla sezione selezionata
if section == "Dataset Information":
    section_introduction()
elif section == "Caricamento dei Dati":
    section_data_loading()
elif section == "Analisi dei Dati":
    section_data_analysis()
elif section == "Risultati e Conclusioni":
    section_results_conclusions()
