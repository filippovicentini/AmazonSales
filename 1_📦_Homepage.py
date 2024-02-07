import streamlit as st
import pandas as pd
from functions import *

st.set_page_config(
    page_title='Amazon Sale Report',
    page_icon=':package:'
)

# Impostare il colore del titolo su blu e arancione
st.title(':blue[Amazon] :orange[Sale] :blue[Report] :package:')
st.sidebar.success('Select a page above', icon='⬆️')

# Testo descrittivo con formattazione Markdown
st.markdown("""
In this web app, we analyze the **Amazon Sale Report** dataset downloaded from Kaggle. The dataset provides information on Amazon sales in India that occurred in April, May and June. Let's explore the data to derive insights and predictions!
""")

filepath = '/Users/filippovicentini/Desktop/programming_project/AmazonSales/datasets/Amazon Sale Report.csv'
no_cleaning_df = pd.read_csv(filepath, low_memory = False)
df = df_cleaning_vis_phase(filepath)

# Sezione per visualizzare le informazioni principali del DataFrame
st.header(':blue[DataFrame] :orange[Information]', divider='orange')
# Widget di selezione per scegliere il DataFrame da visualizzare
df_choice = st.selectbox("Select DataFrame:", ["Original", "Cleaned"])

if df_choice == "Original":
    selected_df = no_cleaning_df
else:
    selected_df = df

st.write(f'**Number of Rows:** {selected_df.shape[0]}')
st.write(f'**Number of Columns:** {selected_df.shape[1]}')

if df_choice == 'Original':
    columns_info = pd.DataFrame({
        'Column name': no_cleaning_df.columns,
        'Description': [
        'Index', 'Order ID', 'Date of the sail',
        'Status of the sail', 'Method of fulfilment', 'Sales Channel',
        'Ship Service Level', 'Style of the product', 'Stock Keeping Unit',
        'Type of product', 'Size of the product', 'Amazon Standard Identification Number',
        'Status of the courier', 'Quantity of the product', 'Currency', 'Amount of the sail',
        'Ship City', 'Ship State', 'Ship Postal Code', 'Ship Country', 'Promotion Ids',
        'Business to business sale', 'fulfilled by', 'Unnamed: 22'
        ],
        'NaN Count': no_cleaning_df.isna().sum().values
    })
    st.table(columns_info)
else:
    columns_info = pd.DataFrame({
    'Column name': df.columns,
    'Description': [
        'Order ID', 'Date of the sail',
        'Status of the sail', 'Method of fulfilment',
        'Ship Service level', 'Style of the product',
        'Stock Keeping unit', 'Type of product',
        'Size of the product', 'Amazon Standard Identification Number',
        'Status of the courier', 'Quantity of the product',
        'Amount of the sale ($)', 'Ship City',
        'Ship State', 'Ship postal code',
        'Promotion IDs', 'Type of the customer', 'Month of the sail'
    ],
    'NaN Count': df.isna().sum().values
    })
    st.table(columns_info)


# Widget di input per inserire l'Order ID
order_id_input = st.text_input("Enter Order ID:", "")

# Mostra la riga corrispondente all'Order ID inserito
if df_choice == 'Original':
    if order_id_input:
        try:
            #order_id_input = int(order_id_input)
            selected_row = no_cleaning_df[no_cleaning_df['Order ID'] == order_id_input]
            st.subheader(f"Details for Order ID: {order_id_input}")
            st.write(selected_row)
        except ValueError:
            st.warning("Please enter a valid Order ID.")
else:
    if order_id_input:
        try:
            #order_id_input = int(order_id_input)
            selected_row = df[df['order_ID'] == order_id_input]
            st.subheader(f"Details for Order ID: {order_id_input}")
            st.write(selected_row)
        except ValueError:
            st.warning("Please enter a valid Order ID.")
