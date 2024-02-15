import streamlit as st
from functions import *
import altair as alt

filepath = '/Users/filippovicentini/Desktop/programming_project/AmazonSales/datasets/Amazon Sale Report.csv'
df = df_cleaning_vis_phase(filepath)

st.title(':blue[Data] :orange[Visualization] :bar_chart:')
st.set_option('deprecation.showPyplotGlobalUse', False)

# Widget per mostrare il grafico della net revenue
show_net_revenue_chart = st.sidebar.radio('Select Chart :arrow_down:', ['Monthly Order Quantity Trend for Category', 'Sales over time', 'Amazon Net Revenue', 'Average Monthly Order Amount', 'Top Product Revenue by Month', 'Sales by Product Size'])
if show_net_revenue_chart == 'Amazon Net Revenue':
    st.subheader(':blue[Amazon Net Revenue]', divider='orange')
    amazon_net_revenue_strim(filepath)
    st.markdown('''This graph shows the **total income for each month**. Specifically, the database was filtered against the 'month' feature 
                and then the sum was calculated against the 'order_amount_($)' feature. It is possible to observe that net revenue
                in May and June turns out to be below the average revenue. In particular, in May it is **4.7%** below the average
                and in June it is **14.9%** below the average.
                ''')
    revenue_by_month = df.groupby('month')['order_amount_($)'].sum()
    percent_decrease_apr_to_may = (revenue_by_month['april'] - revenue_by_month['may']) / revenue_by_month['april'] * 100
    percent_decrease_may_to_jun = (revenue_by_month['may'] - revenue_by_month['june']) / revenue_by_month['may'] * 100
    st.sidebar.metric(label="Total revenue for April 2022", value=round(revenue_by_month['april'], 2))
    st.sidebar.metric(label="Total revenue for May 2022", value=round(revenue_by_month['may'], 2), delta=-round(percent_decrease_apr_to_may,2))
    st.sidebar.metric(label="Total revenue for June 2022", value=round(revenue_by_month['june'], 2), delta = -round(percent_decrease_may_to_jun,2))
    



# Widget per mostrare il grafico dell'average monthly order amount
elif show_net_revenue_chart == 'Average Monthly Order Amount':
    st.subheader(':blue[Average Monthly Order Amount]', divider='orange')
    interactive_average_monthly_order_amount(filepath)
    st.markdown('''
                This graph represents the **average monthly order amount**. In particular, it shows
                that the average monthly order amount is increased of **6.32%** from April to May and of **6%** from April to June.
                ''')
    monthly_aov = df.groupby(pd.Grouper(key='date', freq='M')).agg({'order_amount_($)': 'sum', 'order_ID': 'nunique'})
    monthly_aov['average_order_value'] = monthly_aov['order_amount_($)'] / monthly_aov['order_ID']
    monthly_aov['pct_change'] = monthly_aov['average_order_value'].pct_change() * 100
    pct_change_april_may = ((monthly_aov['average_order_value'][1] - monthly_aov['average_order_value'][0]) / monthly_aov['average_order_value'][0]) * 100
    pct_change_april_june = ((monthly_aov['average_order_value'][2] - monthly_aov['average_order_value'][0]) / monthly_aov['average_order_value'][0]) * 100
    st.sidebar.metric(label="Average order amount in April 2022", value=round(monthly_aov['average_order_value'][0], 2))
    st.sidebar.metric(label="Average order amount in May 2022", value=round(monthly_aov['average_order_value'][1], 2), delta=round(pct_change_april_may,2))
    st.sidebar.metric(label="Average order amount in June 2022", value=round(monthly_aov['average_order_value'][2], 2), delta=round(pct_change_april_june,2))

# Widget per mostrare il grafico del top product revenue by month
elif show_net_revenue_chart == 'Top Product Revenue by Month':
    st.subheader(':blue[Top Product Revenue by Month]', divider='orange')
    interactive_top_product_revenue_by_month(filepath)
    st.markdown('''
                This chart shows top product revenue by month. From the graph, it can be deduced that the product that contributes the most on revenue is 'set'.
                In particular, if we focus on the 'Western Dress' category, we can see a 49% increase in revenue from April to May and a 33% from April to June.
                ''')
    sales_data = df[df['product_category'].isin(['Western Dress', 'Top', 'kurta', 'Set'])]
    sales_by_month = sales_data.groupby(['month', 'product_category'])['order_amount_($)'].sum().reset_index()
    sales_wd = sales_by_month[sales_by_month['product_category'] == 'Western Dress'].reset_index(drop=True)
    pct_increase_april_may = (sales_wd.loc[1, 'order_amount_($)'] - sales_wd.loc[0, 'order_amount_($)']) / sales_wd.loc[0, 'order_amount_($)'] * 100
    pct_increase_april_june = (sales_wd.loc[2, 'order_amount_($)'] - sales_wd.loc[0, 'order_amount_($)']) / sales_wd.loc[0, 'order_amount_($)'] * 100
    st.sidebar.metric(label="Sales data for Western Dress in April 2022", value=round(sales_wd.loc[0, 'order_amount_($)'], 2))
    st.sidebar.metric(label="Sales data for Western Dress in May 2022", value=round(sales_wd.loc[1, 'order_amount_($)'], 2), delta=round(pct_increase_april_may,2))
    st.sidebar.metric(label="Sales data for Western Dress in June 2022", value=round(sales_wd.loc[2, 'order_amount_($)'], 2), delta=round(pct_increase_april_june,2))
    
# Widget per mostrare il grafico delle sales by product size
elif show_net_revenue_chart == 'Sales by Product Size':
    st.subheader(':blue[Sales by Product Size]', divider='orange')
    interactive_sales_by_product_size(filepath)
    st.markdown('''
                The chart above shows revenues by product size. It is possible to observe
                 that the size that produces the highest revenue is M. This fact may be influenced by these two factors:
                - Number of product with size M sold
                - Average price of  single product with size M
                
                In this context, it can be seen that the best-selling size is precisely M
                ''')
    sales_by_size = df.groupby('size')['order_amount_($)'].sum()
    quantity_by_size = df.groupby('size')['order_quantity'].sum()
    col1, col2 = st.sidebar.columns(2)
    col1.metric(label='Net Revenue of size S', value=round(sales_by_size['S']))
    col1.metric(label='Net Revenue of size M', value=round(sales_by_size['M']))
    col1.metric(label='Net Revenue of size L', value=round(sales_by_size['L']))
    col2.metric(label='Quantity of size S sold', value = quantity_by_size['S'])
    col2.metric(label='Quantity of size M sold', value = quantity_by_size['M'])
    col2.metric(label='Quantity of size L sold', value = quantity_by_size['L'])
    interactive_quantity_size(filepath)
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

    st.markdown('This chart shows the **monthly order quantity trend** for the selected category in the selected month')
elif show_net_revenue_chart == 'Sales over time':
    st.subheader(':blue[Sales over time]', divider='orange')
    highlight_option = st.selectbox('Choose what you want to see:', ['None', 'Maximum', 'Minimum', 'Mean'])
    sales_over_time(filepath, highlight_option)
    st.markdown("""
                This graph shows daily revenue trends. 
                In particular, we can derive the respective days of maximum and minimum revenues:
                - **Max: 2022-05-04**
                - **Min: 2022-06-29**

                Also, if you look at the graph with in addition the option to view the average line,
                 you can see that since May 8 there is a majority of revenues that are below average.
                """)
    





