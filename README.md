# AmazonSales
The project consists of an analysis performed on the dataset Amazon Sale Report downloading from kaggle (https://www.kaggle.com/datasets/thedevastator/unlock-profits-with-e-commerce-sales-data?select=Amazon+Sale+Report.csv). The dataset reports some informations about Amazon sales in India in April, May and June 2022. To view the streamlit presentation run the file 1_🏠_Homepage.py.

The project is divided mainly into three parts:

## Data Cleaning
- In this phase the df is prepared for the Visualization phase and the Modeling phase:
  - Columns to drop: 
    - Unnamed: 22 - undeterminable data, 
    - fulfilled-by - only value was amazon courier "easy-ship" with no other relationship, 
    - ship-country - The shipping Country is India, 
    - currency - the currency is Indian Rupee (INR),
    - Sales Channel - assumed to be sold through amazon
  - Date Range - April 1, 2022 to June 29, 2022
  - Columns dropduplicates()
  - Columns fillna():
    - Courier Status - will fill missing with 'Unknown'
    - promotion-ids - will fill missing with 'No Promotion'
    - Amount - will fill missing with 0, since 97% of all Orders with missing Amount are cancelled
    - ship-city, ship-state and ship-postal-code
  - Column Renaming and changing values:
    - B2B - changing to customer_type and changing the values to business and customer
    - Amount - changing to order_amount and converting from INR to $
    - date - dropping march from the datset since there is only 1 day (3/21/22) from the month representing 0.1%
  - Column Creation
    - month - to use in analysis and groupbys
  - Column Value Ordering
    - size - created an ordered category of based on product sizes
## Data Visualization
- The project implements different charts:
    - Amazon Net Revenue
    - Average Monthly Order Amount 
    - Top Product Revenue by Month
    - Sales by Product Size
    - Sales Over Time
    - Monthly Order Quantity Trend for Category
## Modeling
- We shall make ORDER REJECTION CLASSIFICATION PREDICTION using two models based on:
  - Logistic Regression Classifier
  - Random Forest Classifier

In addition, we would like to try to find out the features that most influence this event and also make a comparison between Logistic Regression and Random Forest Classifier.
