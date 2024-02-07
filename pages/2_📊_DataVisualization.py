import streamlit as st
from functions import *
import altair as alt

filepath = '/Users/filippovicentini/Desktop/programming_project/AmazonSales/datasets/Amazon Sale Report.csv'
df = df_cleaning_vis_phase(filepath)

st.title(':blue[Data] :orange[Visualization] :bar_chart:')
st.set_option('deprecation.showPyplotGlobalUse', False)

# Barra laterale per i widget fissi in alto
st.sidebar.title(':arrow_down: Select Chart')

# Widget per mostrare il grafico della net revenue
show_net_revenue_chart = st.sidebar.radio("Select Chart", ['Amazon Net Revenue', 'Average Monthly Order Amount', 'Top Product Revenue by Month', 'Sales by Product Size', 'Bar Chart: Orders by Product'])
if show_net_revenue_chart == 'Amazon Net Revenue':
    st.subheader('Amazon Net Revenue', divider='orange')
    amazon_net_revenue_strim(filepath)

# Widget per mostrare il grafico dell'average monthly order amount
elif show_net_revenue_chart == 'Average Monthly Order Amount':
    st.subheader('Average Monthly Order Amount', divider='orange')
    interactive_average_monthly_order_amount(filepath)

# Widget per mostrare il grafico del top product revenue by month
elif show_net_revenue_chart == 'Top Product Revenue by Month':
    st.subheader('Top Product Revenue by Month', divider='orange')
    interactive_top_product_revenue_by_month(filepath)

# Widget per mostrare il grafico delle sales by product size
elif show_net_revenue_chart == 'Sales by Product Size':
    st.subheader('Sales by Product Size', divider='orange')
    interactive_sales_by_product_size(filepath)

# Widget per mostrare il grafico a barre delle orders by product
elif show_net_revenue_chart == 'Bar Chart: Orders by Product':
    st.subheader('Bar Chart: Orders by Product', divider='orange')
    selected_box = st.selectbox('choose', ['1', '2', '3'])
    if selected_box == '1':
        st.write('ciao')
    if selected_box == '2':
        st.write('no')
    if selected_box == '3':
        st.write('si')


