import streamlit as st
from functions import *
import altair as alt

filepath = '/Users/filippovicentini/Desktop/programming_project/AmazonSales/datasets/Amazon Sale Report.csv'
df = df_cleaning_vis_phase(filepath)

st.title(':blue[Data] :orange[Visualization] :bar_chart:')
st.set_option('deprecation.showPyplotGlobalUse', False)

# Widget per mostrare il grafico della net revenue
show_net_revenue_chart = st.sidebar.radio('Select Chart :arrow_down:', ['Monthly Order Quantity Trend for Category','Amazon Net Revenue', 'Average Monthly Order Amount', 'Top Product Revenue by Month', 'Sales by Product Size'])
if show_net_revenue_chart == 'Amazon Net Revenue':
    st.subheader(':blue[Amazon Net Revenue]', divider='orange')
    amazon_net_revenue_strim(filepath)

# Widget per mostrare il grafico dell'average monthly order amount
elif show_net_revenue_chart == 'Average Monthly Order Amount':
    st.subheader(':blue[Average Monthly Order Amount]', divider='orange')
    interactive_average_monthly_order_amount(filepath)

# Widget per mostrare il grafico del top product revenue by month
elif show_net_revenue_chart == 'Top Product Revenue by Month':
    st.subheader(':blue[Top Product Revenue by Month]', divider='orange')
    interactive_top_product_revenue_by_month(filepath)

# Widget per mostrare il grafico delle sales by product size
elif show_net_revenue_chart == 'Sales by Product Size':
    st.subheader(':blue[Sales by Product Size]', divider='orange')
    interactive_sales_by_product_size(filepath)

# Widget per mostrare il grafico a barre delle orders by product
elif show_net_revenue_chart == 'Monthly Order Quantity Trend for Category':
    selected_category = st.selectbox("Select Product Category", df['product_category'].unique())
    selected_month = st.selectbox('Select Month',  df['month'].unique())
    st.subheader(f':blue[Monthly Order Quantity Trend for {selected_category} in {selected_month}]', divider='orange')

    # Filtra il DataFrame per la categoria e il mese selezionati
    filtered_df = df[(df['product_category'] == selected_category) & (df['month'] == selected_month)]

    # Calcola la somma delle quantità degli ordini per ogni mese e categoria
    monthly_quantity = filtered_df.groupby('date')['order_quantity'].sum().reset_index()
    #st.write(monthly_quantity)
    # Esempio di grafico Altair basato sull'andamento delle quantità degli ordini nel tempo per una categoria specifica
    chart = alt.Chart(monthly_quantity).mark_line(color='#1e90ff').encode(
        x='date:T',
        y='order_quantity:Q',
        tooltip=['date:T', 'order_quantity:Q']
    ).properties(
        width=800,
        height=500,
        #title=f'Monthly Order Quantity Trend for Category: {selected_category} in {selected_month}'
    )

    # Visualizza il grafico Altair
    st.altair_chart(chart, use_container_width=True)





