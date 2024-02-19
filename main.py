import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from functions import *

#loading dataset
df = pd.read_csv('/Users/filippovicentini/Desktop/programming_project/AmazonSales/datasets/Amazon Sale Report.csv')
filepath = '/Users/filippovicentini/Desktop/programming_project/AmazonSales/datasets/Amazon Sale Report.csv'

#this file is used only to show initial dataset view and to compute preliminary insights. 
#To see the full project and the streamlit presentation run the file 1_🏠_Homepage.py

#initial dataset view
print(df.head())
print()
print(df.info())
print()
print(df.nunique().to_frame(name='Count of unique values'))
print()
print(df.apply(pd.unique).to_frame(name='Unique Values'))
print()
print(df.describe().T)
print(df.isnull().sum())

"""
DATA CLEANING:
- Columns to drop: 
    Unnamed: 22 - undeterminable data, 
    fulfilled-by - only value was amazon courier "easy-ship" with no other relationship, 
    ship-country - The shipping Country is India, 
    currency - the currency is Indian Rupee (INR),
    Sales Channel - assumed to be sold through amazon
- Date Range - April 1, 2022 to June 29, 2022
- Columns dropduplicates()
- Columns fillna():
    Courier Status - will fill missing with 'Unknown'
    promotion-ids - will fill missing with 'No Promotion'
    Amount - will fill missing with 0, since 97% of all Orders with missing Amount are cancelled
    ship-city, ship-state and ship-postal-code
- Column Renaming and changing values:
    B2B - changing to customer_type and changing the values to business and customer
    Amount - changing to order_amount and converting from INR to $
    date - dropping march from the datset since there is only 1 day (3/21/22) from the month representing 0.1%
- Column Creation
    month - to use in analysis and groupbys
- Column Value Ordering
    size - created an ordered category of based on product sizes
"""
df = df_cleaning_vis_phase(filepath)

#DATA VISUALIZATION:
#Preliminary Insight
revenue_by_month = df.groupby('month', observed=False)['order_amount_($)'].sum()
percent_decrease_apr_to_may = (revenue_by_month['april'] - revenue_by_month['may']) / revenue_by_month['april'] * 100
percent_decrease_may_to_jun = (revenue_by_month['may'] - revenue_by_month['june']) / revenue_by_month['may'] * 100
total_decrease = (revenue_by_month['april'] - revenue_by_month['june']) / revenue_by_month['april'] * 100
print(f"Total revenue for April 2022: ${revenue_by_month['april']:,.2f}")
print(f"Total revenue for May 2022: ${revenue_by_month['may']:,.2f}, which is a -{percent_decrease_apr_to_may:.2f}% decrease from April.")
print(f"Total revenue for June 2022: ${revenue_by_month['june']:,.2f}, which is a -{percent_decrease_may_to_jun:.2f}% decrease from May.")
print(f"Total revenue for Q2 2022 decreased by -{total_decrease:.2f}%")
print("\n")

revenue_by_category = df.groupby('product_category')['order_amount_($)'].sum().sort_values(ascending=False)
print("Total revenue by product category:")
print(revenue_by_category.apply(lambda x: "${:,.2f}".format(x)))
print("\n")

revenue_by_category = df.groupby('product_category')['order_amount_($)'].sum()
percent_revenue_by_category = ((revenue_by_category / revenue_by_category.sum()) * 100).sort_values(ascending=False)
percent_revenue_by_category = percent_revenue_by_category.apply(lambda x: "{:.2f}%".format(x))
print("Percentage of revenue by product category:")
print(percent_revenue_by_category)
print("\n")

avg_price_by_category = df.groupby('product_category')['order_amount_($)'].mean()
avg_price_by_category = avg_price_by_category.sort_values(ascending=False)
print("Top 5 product categories by average price:")
print(avg_price_by_category.head(5))
print("\n")

cancelled_orders = df[df['ship_status'].isin(['Cancelled', 'Shipped - Lost in Transit'])]
returned_orders = df[df['ship_status'].isin(['Shipped - Returned to Seller', 'Shipped - Returning to Seller', 'Shipped - Rejected by Buyer', 'Shipped - Damaged'])]
total_cancelled = len(cancelled_orders)
total_returned = len(returned_orders)
total_cancelled_returned = total_cancelled + total_returned
percent_cancelled = total_cancelled / len(df) * 100
percent_returned = total_returned / len(df) * 100
percent_cancelled_returned = total_cancelled_returned / df['order_quantity'].sum() * 100
print(f"Total cancelled orders: {total_cancelled}, which is {percent_cancelled:.2f}% of all orders.")
print(f"Total returned orders: {total_returned}, which is {percent_returned:.2f}% of all orders.")
print(f"This represents {percent_cancelled_returned:.2f}% of all orders.")
print("\n")


monthly_order_data = df.groupby(pd.Grouper(key='date', freq='M')).agg({'order_amount_($)': 'mean', 'order_quantity': 'mean'})
monthly_order_data = monthly_order_data.rename(columns={'order_amount_($)': 'average_order_amount', 'order_quantity': 'average_order_quantity'})
print(monthly_order_data)
print("\n")

popular_category_by_state = df.groupby(['state', 'product_category'])['order_quantity'].sum().reset_index()
popular_category_by_state = popular_category_by_state.sort_values(['state', 'order_quantity'], ascending=[True, False])
popular_category_by_state = popular_category_by_state.drop_duplicates(subset=['state'])
print("Most popular product category in each state:")
print(popular_category_by_state)
print("\n")

avg_order_amount_by_customer_type = df.groupby('customer_type')['order_amount_($)'].mean()
print("Average order amount by customer type:")
print(avg_order_amount_by_customer_type.apply(lambda x: "${:,.2f}".format(x)))

