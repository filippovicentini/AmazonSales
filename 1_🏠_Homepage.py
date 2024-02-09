import streamlit as st
import pandas as pd
from functions import *

st.set_page_config(
    page_title='Amazon Sale Report',
    page_icon=':truck:'
)

# Impostare il colore del titolo su blu e arancione
st.title(':blue[Amazon] :orange[Sale] :blue[Report] :truck:')
st.sidebar.success('Select a page above', icon='⬆️')

# Testo descrittivo con formattazione Markdown
st.markdown("""
In this web app, we analyze the **Amazon Sale Report** dataset downloaded from Kaggle. 
The dataset provides information on Amazon sales in India that occurred in April, May, and June. 
Let's explore the data to derive insights and predictions! :bar_chart:""")

filepath = '/Users/filippovicentini/Desktop/programming_project/AmazonSales/datasets/Amazon Sale Report.csv'
no_cleaning_df = pd.read_csv(filepath, low_memory = False)
df = df_cleaning_vis_phase(filepath)

# Sezione per visualizzare le informazioni principali del DataFrame
st.header(':blue[DataFrame] :orange[Information]', divider='orange')
# Widget di selezione per scegliere il DataFrame da visualizzare
df_choice = st.selectbox("Select DataFrame:", ["Cleaned","Original"])

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

# Mostra le informazioni corrispondenti all'Order ID inserito
if df_choice == 'Cleaned':
    # Widget di input per inserire l'Order ID nella barra laterale
    order_id_input = st.sidebar.text_input("Enter Order ID:", "")
    if order_id_input:
        try:
            selected_row = df[df['order_ID'] == order_id_input]
            if not selected_row.empty:
                st.sidebar.subheader(f"Details for Order ID: {order_id_input}")

                # Seleziona solo le colonne desiderate
                selected_info = selected_row[['date', 'product_category',
                                              'size', 'courier_ship_status', 'order_quantity',
                                              'order_amount_($)', 'city', 'state']]

                # Formatta le informazioni come una lista di stringhe
                formatted_info = [f"{col}: {value}" for col, value in selected_info.to_dict(orient='records')[0].items()]

                # Mostra le informazioni nella barra laterale
                for info in formatted_info:
                    st.sidebar.text(info)
            else:
                st.sidebar.warning("No details found for the provided Order ID.")
        except ValueError:
            st.sidebar.warning("Please enter a valid Order ID.")



