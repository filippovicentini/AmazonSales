#import libraries and functions
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,roc_curve,roc_auc_score
from sklearn import set_config
set_config(display = "diagram")  
from functions import *

#loading dataset
df = pd.read_csv('/Users/filippovicentini/Desktop/programming_project/AmazonSales/datasets/Amazon Sale Report.csv')

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
where_nan(df)
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
#dropping columns
df.drop(columns= ['index','Unnamed: 22', 'fulfilled-by', 'ship-country', 
                  'currency', 'Sales Channel '], inplace = True)

#dropping duplicates
#df[df.duplicated(['Order ID','ASIN'], keep=False)]
#len(df)-len(df.drop_duplicates(['Order ID','ASIN']))
df.drop_duplicates(['Order ID','ASIN'],inplace = True,ignore_index=True)

#filling NaN values
df['Courier Status'].fillna('unknown',inplace=True)
df['promotion-ids'].fillna('no promotion',inplace=True)
print()
print(df[df['Amount'].isnull()]['Status'].value_counts(normalize=True).apply(lambda x: format(x, '.2%')))
print()
df['Amount'].fillna(0,inplace=True)
df['ship-city'].fillna('unknown', inplace = True)
df['ship-postal-code'] = df['ship-postal-code'].astype(str)
df['ship-postal-code'].fillna('unknown', inplace=True)
df['ship-state'].fillna('unknown', inplace = True)

#renaming columns
mapper = {'Order ID':'order_ID', 'Date':'date', 'Status':'ship_status','Fulfilment':'fullfilment',
          'ship-service-level':'service_level', 'Style':'style', 'SKU':'sku', 'Category':'product_category', 
          'Size':'size', 'ASIN':'asin', 'Courier Status':'courier_ship_status', 'Qty':'order_quantity', 
          'Amount':'order_amount_($)', 'ship-city':'city', 'ship-state':'state', 'ship-postal-code':'zip', 
          'promotion-ids':'promotion','B2B':'customer_type'}
df.rename(columns=mapper, inplace =True)

#Convert INR to USD using an exchange rate of 1 INR = 0.014 USD
exchange_rate = 0.0120988
df['order_amount_($)'] = df['order_amount_($)'].apply(lambda x: x * exchange_rate)

#Convert B2B column values
df['customer_type'].replace(to_replace=[True,False],value=['business','customer'], inplace=True)

#Creating Datetime and adding Month column
df['date'] = pd.to_datetime(df['date'], format='%m-%d-%y')
# Filter to only include dates in March
march_dates = df['date'][df['date'].dt.month == 3]
# Get the number of unique days in March
march_dates.dt.day.nunique()
# dropping March dates from the dataset
df = df[(df['date'].dt.month != 3)]
df['month'] = df['date'].dt.month
df["month"].unique()
month_map = { 4: 'april',5: 'may',6: 'june'}
df['month'] = df['date'].dt.month.map(month_map)
# Define the desired order of months
month_order = ['april', 'may', 'june']
# Convert the month column to a categorical data type with the desired order
df['month'] = pd.Categorical(df['month'], categories=month_order, ordered=True)
print(f'This dataset contains the months {df["month"].unique()} for 2022')
print(f'The earliest date is {df["date"].min()}')
print(f'The latest date is {df["date"].max()}')
#Column Value Ordering
# Define the desired order for the 'size' column
size_order = ['Free','XS', 'S', 'M', 'L', 'XL', 'XXL', '3XL', '4XL', '5XL', '6XL']
# Create an ordered categorical variable for the 'size' column
df['size'] = pd.Categorical(df['size'], categories=size_order, ordered=True)

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

#Show amazon net revenue
amazon_net_revenue(df)

#Show average monthly order amount
average_monthly_order_amount(df)

#Show top product revenue by month
top_product_revenue_by_month(df)

#Show sales by product size
sales_by_product_size(df)

#Show heatmap of quantity sold by Category and Size
heatmap_category_size(df)

#Show top 10 cities with the most orders
top_cities(df)

#Order rejection classification
order_rejection(df)