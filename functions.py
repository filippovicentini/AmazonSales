import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,roc_curve,roc_auc_score,auc
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn import set_config
import altair as alt
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB

set_config(display = "diagram")  

#FUNCTION FOR DATA VISUALIZATION:
#this function shows amazon net revenue
def amazon_net_revenue_strim(filepath):
    df = df_cleaning_vis_phase(filepath)
    sns.set_style('whitegrid')

    # Group the data by month and calculate the total sales revenue
    monthly_sales = df.groupby(pd.Grouper(key='date', freq='M')).agg({'order_amount_($)': 'sum'})

    # Get latest month revenue and average quarterly revenue
    latest_month_revenue = monthly_sales.tail(1).iloc[0][0]
    avg_quarterly_revenue = monthly_sales.tail(3).head(2).mean()[0]

    # Compute percentage below average revenue for quarter
    pct_below_avg = round((1 - (latest_month_revenue / avg_quarterly_revenue)) * 100, 1)

    # Plot the monthly sales revenue
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(monthly_sales.index.strftime('%b'), monthly_sales['order_amount_($)'], 
                  color='#1e90ff')

    # Add label above each bar with the percentage below the average revenue for the quarter
    for i, bar in enumerate(bars):
        if i == len(bars) - 1 or i < len(bars) - 2:
            continue
        month_sales = monthly_sales.iloc[i]['order_amount_($)']
        pct_below_avg = round((1 - (month_sales / avg_quarterly_revenue)) * 100, 1)
        ax.annotate(f'{pct_below_avg}% below avg.', 
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()-7000), 
                    xytext=(0, 5), textcoords='offset points',  fontweight='bold', 
                    ha='center', va='bottom', fontsize=14)

    # Add label above the latest bar with the percentage below the average revenue for the quarter
    latest_bar = bars[-1]
    latest_month_sales = latest_bar.get_height()
    pct_below_avg = round((1 - (latest_month_sales / avg_quarterly_revenue)) * 100, 1)
    ax.annotate(f'{pct_below_avg}% below avg.', 
                xy=(latest_bar.get_x() + latest_bar.get_width()/2, latest_bar.get_height()-7000), 
                xytext=(0, 5), textcoords='offset points',  fontweight='bold',
                ha='center', va='bottom', fontsize=14)

    # Add horizontal line at the average quarterly revenue
    plt.axhline(avg_quarterly_revenue, linestyle='--', color='orange',linewidth=2, label='Average Revenue')

    #ax.set_title('Amazon India Net Revenue', fontsize=20, x=.19, y=1.05)
    #ax.text(-.08, 1.02, 'Q2 FY22', fontsize=15, color='#878787', transform=ax.transAxes)
    ax.set_xlabel(None)
    ax.set_yticklabels(list(range(0,41,5)))
    ax.set_ylabel('Net Revenue in 10,000 dollars', fontsize=12, labelpad=3)

    ax.yaxis.grid(linestyle='--', color='gray', linewidth=0.5, dashes=(8, 5))
    ax.xaxis.grid(False)
    plt.legend(bbox_to_anchor=(1,1.05), fontsize=12, fancybox=True)

    ax.tick_params(axis='both', labelsize=12)
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_color('black')

    st.pyplot(fig)

#this function shows average monthly order amount
def interactive_average_monthly_order_amount(filepath):
    # Pulizia del DataFrame
    df = df_cleaning_vis_phase(filepath)

    # Gruppo dei dati per mese e calcolo del valore medio dell'ordine
    monthly_aov = df.groupby(pd.Grouper(key='date', freq='M')).agg({'order_amount_($)': 'sum', 'order_ID': 'nunique'})
    monthly_aov['average_order_value'] = monthly_aov['order_amount_($)'] / monthly_aov['order_ID']

    # Calcolo della variazione percentuale dal mese precedente
    monthly_aov['pct_change'] = monthly_aov['average_order_value'].pct_change() * 100

    # Creazione di un grafico a barre del valore medio dell'ordine per mese
    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(x=monthly_aov.index.strftime('%b'), y=monthly_aov['average_order_value'], ax=ax, color='#1e90ff')

    # Aggiunta di un grafico a linee del valore medio dell'ordine per mese
    ax.plot(monthly_aov.index.strftime('%b'), monthly_aov['average_order_value'], linestyle='--', linewidth=2, color='orange', marker='o')

    # Aggiunta del riferimento percentuale dall'incremento da aprile a giugno
    apr_val = monthly_aov['average_order_value'].iloc[0]
    jun_val = monthly_aov['average_order_value'].iloc[2]
    pct_change = ((jun_val - apr_val) / apr_val) * 100
    ax.annotate(f'Increase of {pct_change:.2f}% from Apr to Jun', fontweight='bold', xy=(2, 8.074941567466606),
                xytext=(1.65, 8.264941567466606), fontsize=13, ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5, connectionstyle="arc3,rad=-0.1"))

    # Impostazione di etichette e titolo
    #ax.set_title('Average Monthly Order Amount', fontsize=20, x=.22, y=1.07)
    #ax.text(-0.09, 1.04, 'Q2 FY22', fontsize=15, color='#878787', transform=ax.transAxes)
    ax.set_xlabel(None)
    ax.set_ylabel('Average Order Value ($)', fontsize=12, labelpad=3)
    ax.set_ylim(7.20, 8.50)
    ax.yaxis.grid(linestyle='--', color='gray', linewidth=0.5, dashes=(8, 5))

    ax.tick_params(axis='both', labelsize=12)
    # Rimozione delle spine superiori e laterali
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_color('black')

    # Utilizzo di Streamlit per mostrare il grafico
    st.pyplot(fig)

#this function shows the top product revenue by month
def interactive_top_product_revenue_by_month(filepath):
    import warnings
    warnings.filterwarnings('ignore')
    
    # Pulizia del DataFrame
    df = df_cleaning_vis_phase(filepath)
    
    # Creazione di un oggetto figura e assi
    fig, ax = plt.subplots(figsize=(8,6))

    # Definizione dell'ordine desiderato dei mesi
    month_order = ['April', 'May', 'June']

    # Filtraggio dei dati includendo solo le quattro categorie di prodotti di interesse
    sales_data = df[df['product_category'].isin(['Western Dress', 'Top', 'kurta', 'Set'])]

    # Conversione della colonna delle date in un oggetto datetime
    sales_data['date'] = pd.to_datetime(sales_data['date'])

    # Estrazione del mese dalla colonna delle date e impostazione come nuova colonna
    sales_data['month'] = sales_data['date'].dt.month_name()

    # Aggregazione dei dati di vendita per mese e categoria di prodotto
    sales_by_month = sales_data.groupby(['month', 'product_category'])['order_amount_($)'].sum().reset_index()

    # Conversione della colonna del mese in un tipo di dato categorico con l'ordine desiderato
    sales_by_month['month'] = pd.Categorical(sales_by_month['month'], categories=month_order, ordered=True)

    # Creazione del grafico delle vendite usando seaborn
    ax = sns.barplot(x='month', y='order_amount_($)', hue='product_category', data=sales_by_month,
                     palette=['#1e90ff', 'grey', 'orange', '#d9d9d9'])

    # Estrazione dei dati di vendita per Western Dress
    sales_wd = sales_by_month[sales_by_month['product_category'] == 'Western Dress'].reset_index(drop=True)
    sales_wd['month'] = pd.Categorical(sales_wd['month'], categories=month_order, ordered=True)
    sales_wd.sort_values(by='month', inplace=True)
    
    # Aggiunta di un grafico a linee per il totale delle entrate mensili di Western Dress
    ax.plot([0.1, 1.1, 2.1], sales_wd['order_amount_($)'], color='black', linestyle='--', linewidth=2, marker='o')

    # Aggiunta annotazione per l'incremento percentuale da aprile a giugno per Western Dress
    pct_increase = (sales_wd.loc[1, 'order_amount_($)'] - sales_wd.loc[0, 'order_amount_($)']) / sales_wd.loc[0, 'order_amount_($)'] * 100
    ax.annotate(f'{pct_increase:.0f}% increase April to June', fontweight='bold', xy=(2.1, sales_wd.loc[2, 'order_amount_($)']),
                xytext=(1.88, sales_wd.loc[2, 'order_amount_($)'] + 40000), arrowprops=dict(arrowstyle='->', color='black', lw=1.5, connectionstyle="arc3,rad=0.1"))

    # Impostazione del numero di tick y desiderati
    num_y_ticks = 10

    # Calcolo dei valori tick y
    max_y_value = sales_by_month['order_amount_($)'].max()
    min_y_value = 0  # Puoi impostare il valore minimo in base ai tuoi dati

    # Calcolo dei valori tonde degli y ticks
    rounded_y_ticks = np.linspace(min_y_value, max_y_value, num_y_ticks).round(-4)

    # Impostazione dei tick y
    ax.set_yticks(rounded_y_ticks)

    # Aggiunta titolo e etichette degli assi
    #ax.set_title('Top Product Revenue by Month', fontsize=20, x=.22, y=1.07)
    #ax.text(-0.09, 1.04, 'Q2 FY22', fontsize=15, color='#878787', transform=ax.transAxes)

    plt.legend(bbox_to_anchor=(1, 1), fontsize=12, framealpha=1)

    ax.set_xlabel(None)
    ax.set_ylabel('Net Revenue', fontsize=12, labelpad=3)
    #ax.set_yticklabels(list(range(0, 46, 5)))
    ax.yaxis.grid(linestyle='--', color='gray', linewidth=0.5, dashes=(8, 5))

    ax.tick_params(axis='both', labelsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_color('black')

    # Utilizzo di Streamlit per mostrare il grafico
    st.pyplot(fig)

    warnings.filterwarnings('default')  # Re-enable the warnings

#this function shows sales by product size
def interactive_sales_by_product_size(filepath):
    # Cleaning the DataFrame
    df = df_cleaning_vis_phase(filepath)

    # Grouping data by product size and calculating total sales
    sales_by_size = df.groupby('size')['order_amount_($)'].sum()

    # Creating a horizontal bar chart to show sales by product size
    fig, ax = plt.subplots(figsize=(12, 6))

    # Using a color palette to highlight specific sizes
    palette_colors = ['orange' if size in ['S', 'M', 'L'] else '#1e90ff' for size in sales_by_size.index]
    sns.barplot(x=sales_by_size.index, y=sales_by_size.values, ax=ax, palette=palette_colors)

    # Setting font sizes for x and y labels, title, and ticks
    ax.set_xlabel('Product Size', labelpad=3, fontsize=14)
    ax.set_ylabel('Net Revenue in 10,000 dollars', labelpad=3, fontsize=14)
    ax.set_yticklabels(list(range(0, 20, 2)))
    #ax.set_title('Sales by Product Size', fontsize=20, x=0.085, y=1.05, pad=10)
    #ax.text(-0.06, 1.04, 'Q2 FY22', fontsize=15, color='#878787', transform=ax.transAxes)

    ax.tick_params(axis='both', labelsize=12)
    ax.yaxis.grid(linestyle='--', color='gray', linewidth=0.5, dashes=(8, 5))
    ax.xaxis.grid(False)

    # Setting the number of y ticks desired
    num_y_ticks = 10

    # Calculating the y tick values
    y_tick_values = np.linspace(ax.get_yticks()[0], ax.get_yticks()[-1], num_y_ticks)

    # Setting the y ticks
    ax.set_yticks(y_tick_values)

    # Setting font sizes for the bars and adding annotations for S, M, and L sizes
    for i, size in enumerate(sales_by_size.index):
        if size in ['S', 'M', 'L']:
            ax.text(i, sales_by_size.values[i], f'{sales_by_size.values[i]/10000:.0f}k', ha='center', fontsize=14, fontweight='bold', color='black')

    # Removing top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_color('black')

    # Using Streamlit to display the plot
    st.pyplot(fig)

#this function shows sales over time
def sales_over_time(filepath, highlight_option):
    df = df_cleaning_vis_phase(filepath)
    
    # Gruppo per data e calcola la somma delle vendite
    sale_over_time = df.groupby('date')['order_amount_($)'].sum().reset_index()

    # Utilizza Altair per creare un grafico a linee
    chart = alt.Chart(sale_over_time).mark_line().encode(
        x='date:T',
        y=alt.Y('order_amount_($):Q', title='Net Revenue ($)')
    ).properties(
        width=800,
        height=400
    )

    # Aggiungi le linee per il massimo, il minimo e la media
    if highlight_option == 'Maximum':
        max_value = sale_over_time.loc[sale_over_time['order_amount_($)'].idxmax()]
        max_line = alt.Chart(pd.DataFrame({'date': [max_value['date']], 'order_amount_($)': [max_value['order_amount_($)']]})).mark_rule(color='red').encode(
            x='date:T',
            y='order_amount_($):Q'
        )
        chart += max_line
        st.sidebar.metric(label='Max on '+max_value['date'].strftime('%Y-%m-%d'), value=round(max_value[1],2))
    elif highlight_option == 'Minimum':
        min_value = sale_over_time.loc[sale_over_time['order_amount_($)'].idxmin()]
        min_line = alt.Chart(pd.DataFrame({'date': [min_value['date']], 'order_amount_($)': [min_value['order_amount_($)']]})).mark_rule(color='blue').encode(
            x='date:T',
            y='order_amount_($):Q'
        )
        chart += min_line
        st.sidebar.metric(label='Min on '+min_value['date'].strftime('%Y-%m-%d'), value=round(min_value[1],2))
    elif highlight_option == 'Mean':
        mean_value = sale_over_time['order_amount_($)'].mean()
        mean_line = alt.Chart(pd.DataFrame({'mean_value': [mean_value]})).mark_rule(color='green').encode(
            y=alt.Y('mean_value:Q', title='Net Revenue ($)')
        )
        chart += mean_line
        st.sidebar.metric(label='Mean', value=round(mean_value,2))

    # Visualizza il grafico utilizzando Streamlit
    st.altair_chart(chart, use_container_width=True)

#this fnction shows the quantity of product sold by size
def interactive_quantity_size(filepath):
    # Cleaning the DataFrame
    df = df_cleaning_vis_phase(filepath)

    # Grouping data by product size and calculating total sales
    sales_by_size = df.groupby('size')['order_quantity'].sum()

    # Creating a horizontal bar chart to show sales by product size
    fig, ax = plt.subplots(figsize=(12, 6))

    # Using a color palette to highlight specific sizes
    palette_colors = ['orange' if size in ['S', 'M', 'L'] else '#1e90ff' for size in sales_by_size.index]
    sns.barplot(x=sales_by_size.index, y=sales_by_size.values, ax=ax, palette=palette_colors)

    # Setting font sizes for x and y labels, title, and ticks
    ax.set_xlabel('Product Size', labelpad=3, fontsize=14)
    ax.set_ylabel('Quantity', labelpad=3, fontsize=14)
    #ax.set_yticklabels(list(range(0, 20, 2)))
    #ax.set_title('Sales by Product Size', fontsize=20, x=0.085, y=1.05, pad=10)
    #ax.text(-0.06, 1.04, 'Q2 FY22', fontsize=15, color='#878787', transform=ax.transAxes)

    ax.tick_params(axis='both', labelsize=12)
    ax.yaxis.grid(linestyle='--', color='gray', linewidth=0.5, dashes=(8, 5))
    ax.xaxis.grid(False)

    # Setting the number of y ticks desired
    num_y_ticks = 10

    # Calculating the y tick values
    y_tick_values = np.linspace(ax.get_yticks()[0], ax.get_yticks()[-1], num_y_ticks)

    # Setting the y ticks
    ax.set_yticks(y_tick_values)

    # Removing top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_color('black')

    # Using Streamlit to display the plot
    st.pyplot(fig)

#FUNCTION FOR DATA CLEANING:
    
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
#this function prepares the dataset for the visualization phase
def df_cleaning_vis_phase(filepath):
    df = pd.read_csv(filepath, low_memory = False)
    #dropping columns
    df.drop(columns= ['index','Unnamed: 22', 'fulfilled-by', 'ship-country', 
                    'currency', 'Sales Channel '], inplace = True)
    df.drop_duplicates(['Order ID','ASIN'],inplace = True,ignore_index=True)
    df['Courier Status'].fillna('unknown',inplace=True)
    df['promotion-ids'].fillna('no promotion',inplace=True)
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
    # dropping March dates from the dataset
    df = df[(df['date'].dt.month != 3)]
    df['month'] = df['date'].dt.month
    month_map = { 4: 'april',5: 'may',6: 'june'}
    df['month'] = df['date'].dt.month.map(month_map)
    # Define the desired order of months
    month_order = ['april', 'may', 'june']
    # Convert the month column to a categorical data type with the desired order
    df['month'] = pd.Categorical(df['month'], categories=month_order, ordered=True)
    # Define the desired order for the 'size' column
    size_order = ['Free','XS', 'S', 'M', 'L', 'XL', 'XXL', '3XL', '4XL', '5XL', '6XL']
    # Create an ordered categorical variable for the 'size' column
    df['size'] = pd.Categorical(df['size'], categories=size_order, ordered=True)
    return df
    
#this function prepare the dataset for the modeling phase
def df_cleaning_modeling_phase(filepath):
    df = pd.read_csv(filepath, low_memory = False)
    #clean the column names
    col = [element.lower().replace(" ","").replace("-","") for element in df.columns]
    df.columns = col
    
    # drop the redundant cols " in the dataset
    df.drop(["index","date","fulfilledby","currency","unnamed:22","promotionids","courierstatus",
             "shipcountry"],axis ="columns", inplace = True)
    
    # fill value "unknown" in rows where location (city and state) is not known 
    df["shipstate"].fillna("unknown", inplace = True)
    df["shipcity"].fillna("unknown", inplace = True)
    
    # fill value 0 in rows where postalcode is null
    df["shippostalcode"].fillna(0, inplace = True)
    
    #change dtype of postalcode to int
    df["shippostalcode"] = df["shippostalcode"].astype(int).astype(object)
    
    # clean the col shipstate to atain a 37 unique values i.e.
    # (28 states + 8 UT + 1 as "UNKNOWN")
    df["shipstate"] = df["shipstate"].str.upper()
    df["shipstate"].replace({"PONDICHERRY":"PUDUCHERRY","RAJSHTHAN":"RAJASTHAN","RAJSTHAN":"RAJASTHAN",
                              "RJ":"RAJASTHAN","PB":"PUNJAB","PUNJAB/MOHALI/ZIRAKPUR":"PUNJAB",
                              "ORISSA":"ODISHA","DELHI":"NEW DELHI","NL":"UNKNOWN","APO":"UNKNOWN",
                              "AR":"UNKNOWN"}, inplace = True)
    
    # drop duplicate rows,if any 
    df.drop_duplicates(inplace = True)
    
    # first drop rows in df with multiple-product orders and then drop column "orderid"
    df = df[df["orderid"].duplicated(keep = False) == False]
    df.drop("orderid", axis = 1, inplace = True)
    
    # drop the rows with unsure rejection status
    known_value = ["Cancelled", 'Shipped - Returned to Seller','Shipped - Rejected by Buyer',
                'Shipped - Returning to Seller','Shipped - Delivered to Buyer']
    df = df[df["status"].isin(known_value)]   

    # create a col "rejected" where value 1 means rejected and 0 means not-rejected" 
    rejected = ["Cancelled", 'Shipped - Returned to Seller','Shipped - Rejected by Buyer',
                'Shipped - Returning to Seller']
    df["rejected"] = df["status"].isin(rejected).astype(int)    # change the dtype to "int"  

    # drop col "status" 
    df.drop("status",axis = "columns", inplace = True)
    
    # drop high cardinality features
    df.drop(["style","sku","shipcity","shippostalcode","asin"],axis = 1, inplace = True)
    
    # drop feature "qty"
    df.drop("qty", axis = "columns", inplace = True)
    
    # replace 0s with "NaN"
    df["amount"] = df["amount"].replace(0,np.nan)
    
    # remove outliers by removing highest 5 percentile of the "amount" feature
    # Note: we still include the rows with value "NaN" 
    df = df[(df["amount"] < df["amount"].quantile(0.95)) | df["amount"].isnull()]
    
    # drop the feature "saleschannel"
    df.drop("saleschannel", axis = "columns", inplace = True)
    
    # Add a col named "regions" based on the geographical location of states
    df["region"] = df["shipstate"].replace({
        "MAHARASHTRA":"westindia","KARNATAKA":"southindia",
        'PUDUCHERRY':"southindia",'TELANGANA':"southindia",
        'ANDHRA PRADESH':"southindia", 'HARYANA':"northindia",
        'JHARKHAND':"eastindia", 'CHHATTISGARH':"eastindia",
        'ASSAM':"northeastindia",'ODISHA':"eastindia",
        'UTTAR PRADESH':"northindia", 'GUJARAT':"westindia",
        'TAMIL NADU':"southindia", 'UTTARAKHAND':"northindia",
        'WEST BENGAL':"eastindia", 'RAJASTHAN':"westindia",
        'NEW DELHI':"centralindia",'MADHYA PRADESH':"centralindia",
        'KERALA':"southindia", 'JAMMU & KASHMIR':"northindia",
        'BIHAR':"eastindia",'MEGHALAYA':"northeastindia",
        'PUNJAB':"northindia", 'GOA':"southindia",
        'TRIPURA':"northeastindia", 'CHANDIGARH':"northindia",
        'HIMACHAL PRADESH':"northindia",'SIKKIM':"northeastindia",
        "ANDAMAN & NICOBAR ":"eastindia", 'MANIPUR':"northeastindia",
        'MIZORAM':"northeastindia",'NAGALAND':"northeastindia",
        'ARUNACHAL PRADESH':"northeastindia", 'LADAKH':"northindia",
        'DADRA AND NAGAR':"westindia",'LAKSHADWEEP':"southindia"
    })

    # drop rows with "UNKNOWN" shipstates
    df = df[df["shipstate"] != "UNKNOWN"]
    
    # drop the feature "shipstate"
    df.drop("shipstate",axis = "columns", inplace = True)
    
    # change the data type of feature "b2b" to object
    df["b2b"] = df["b2b"].astype(object)
    
    # reset index the dataframe
    df = df.reset_index(drop = True)
    
    # return clean dataset 
    return df

#FUNCTION FOR MODELING PHASE:
#this function shows the distribution of the target vector for the training set
def plot_class_distr(y_train):
    # bar chvalue_countsg matplotlib package
    fig,ax = plt.subplots(figsize = (7,5))

    # calculate and store the proportion values in y_train series
    plot_dataseries = round(y_train.value_counts(normalize = True)*100,2) 

    # plot the bar chart
    plot_dataseries.plot(kind = "bar",ax =ax, color = "navy")
    #plt.text(0.5,70,"Imbalanced Class Dataset",color = "darkred",
            #horizontalalignment = "center",fontsize = 14)
    plt.axhline(y = plot_dataseries[0],color = "darkred", linestyle = "--")
    plt.title("Class Proportions of y_train", fontsize = 16)
    plt.ylabel("Proportion", fontsize = 12)
    plt.xlabel("Status", fontsize = 12)
    plt.xticks(ticks = range(len(plot_dataseries)),
            labels = ["Not Rejected", "Rejected"], rotation = "horizontal")
    plt.yticks(ticks = [20,40,60,80,100], labels = ["20%","40%","60%","80%","100%"])

    # create another series with values to be diaplayed as data-label/value-label in the chart
    data_label = plot_dataseries.astype(str).str.cat(np.full((2,),"%"), sep = "")

    # add/plot the data-label (in%) in the chart
    for x,y in enumerate(plot_dataseries):
        plt.text(x,y-7,data_label[x],color = "white",
                fontweight = 700,fontsize = 13, horizontalalignment = "center")

    # add/plot the data-label (in count) in the chart
    for x,y in enumerate(y_train.value_counts()):
        plt.text(x,plot_dataseries[x]- 13,f"Count:{y_train.value_counts()[x]}",
                horizontalalignment = "center", color = "lightpink",fontsize = 12, fontweight = 700)


    st.pyplot(fig)

#this function compute the logistic regression and the random forest classification.
#In particular it returns: confusion matrix for the test set, the roc curve
#and also the top 10 features for each model.
def logistic_and_forest(filepath, size):
    df = df_cleaning_modeling_phase(filepath)
    df['amount'].fillna(0,inplace=True)
    target = "rejected"
    y = df[target]     # target vector
    X = df.drop(target, axis = "columns")   # feature matrix/dataframe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = 42)
    # resample it (training dataset) using Random Over Sampling
    ros = RandomOverSampler(random_state = 42)
    X_train_over,y_train_over = ros.fit_resample(X_train, y_train)
    #LOGISTIC REGRESSION
    num_transformer = make_pipeline(SimpleImputer(),MinMaxScaler())
    cat_transformer = make_pipeline(SimpleImputer(strategy = "most_frequent"),OneHotEncoder(drop = "first"))
    col_transformer = ColumnTransformer(
            [
                ("numtransformer",num_transformer,
                X_train_over.select_dtypes(exclude = "object").columns),
                ("cattransformer",cat_transformer,X_train_over.select_dtypes(include = "object").columns)
            ]
    )
    logistic_model = make_pipeline(
            col_transformer,
            LogisticRegression(random_state = 42, max_iter = 1000)
    )
    logistic_model.fit(X_train_over,y_train_over)

    #training
    y_train_over_pred_lr = logistic_model.predict(X_train_over)
    confusion_train_lr = confusion_matrix(y_train_over, y_train_over_pred_lr)
    accuracy_train_lg = round(logistic_model.score(X_train_over,y_train_over)*100,2)
    precision_train_lg = round(confusion_train_lr[1,1]/(confusion_train_lr[1,1]+confusion_train_lr[0,1])*100,2)
    recall_train_lg = round(confusion_train_lr[1,1]/(confusion_train_lr[1,1]+confusion_train_lr[1,0])*100,2)

    #test
    y_test_pred_lr = logistic_model.predict(X_test)
    confusion_test_lr = confusion_matrix(y_test,y_test_pred_lr)
    accuracy_test_lg = round(logistic_model.score(X_test,y_test)*100,2)
    precision_test_lg = round(confusion_test_lr[1,1]/(confusion_test_lr[1,1]+confusion_test_lr[0,1])*100,2)
    recall_test_lg = round(confusion_test_lr[1,1]/(confusion_test_lr[1,1]+confusion_test_lr[1,0])*100,2)

    #roc curve
    fpr_lr,tpr_lr,thresh_lr = roc_curve(y_test,
                                logistic_model.predict_proba(X_test)[:,1],
                                pos_label = 1)
    auc_lr = roc_auc_score(y_test,logistic_model.predict_proba(X_test)[:,1])

    #futures importance
    # coefficients/weights list
    weights_list = logistic_model.named_steps["logisticregression"].coef_.reshape(-1,)
    # features list
    features_list_lr = list(logistic_model["columntransformer"].transformers_[1][1]["onehotencoder"]
                    .get_feature_names_out())
    features_list_lr.insert(0,"amount")
    #create the pandas series
    feature_importances_lr = pd.Series(weights_list,index = features_list_lr)
    # sort values based on weights 
    feature_importances_lr = feature_importances_lr.sort_values(key = abs)

    #RANDOM FOREST
    X_encoded = pd.get_dummies(X)
    X_trainrf, X_testrf, y_trainrf, y_testrf = train_test_split(X_encoded, y, test_size=size, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_trainrf, y_trainrf)

    #training
    y_train_predrf = model.predict(X_trainrf)
    conf_matrix_train_rf = confusion_matrix(y_trainrf, y_train_predrf)
    accuracy_train_rf = round(model.score(X_trainrf,y_trainrf)*100,2)
    precision_train_rf = round(conf_matrix_train_rf[1,1]/(conf_matrix_train_rf[1,1]+conf_matrix_train_rf[0,1])*100,2)
    recall_train_rf = round(conf_matrix_train_rf[1,1]/(conf_matrix_train_rf[1,1]+conf_matrix_train_rf[1,0])*100,2)

    #test
    y_test_predrf = model.predict(X_testrf)
    conf_matrix_test_rf = confusion_matrix(y_testrf, y_test_predrf)
    accuracy_test_rf = round(model.score(X_testrf,y_testrf)*100,2)
    precision_test_rf = round(conf_matrix_test_rf[1,1]/(conf_matrix_test_rf[1,1]+conf_matrix_test_rf[0,1])*100,2)
    recall_test_rf = round(conf_matrix_test_rf[1,1]/(conf_matrix_test_rf[1,1]+conf_matrix_test_rf[1,0])*100,2)

    #roc curve
    fpr_rf,tpr_rf,thresh_rf = roc_curve(y_testrf,
                            model.predict_proba(X_testrf)[:,1],
                            pos_label = 1)
    auc_rf = roc_auc_score(y_testrf,model.predict_proba(X_testrf)[:,1])

    # Feature Importance
    feature_importances_rf = pd.Series(model.feature_importances_, index=X_encoded.columns)
    feature_importances_rf = feature_importances_rf.sort_values(ascending=True)

    return accuracy_test_lg, precision_test_lg, recall_test_lg, auc_lr, fpr_lr, tpr_lr, feature_importances_lr, accuracy_test_rf, precision_test_rf, recall_test_rf, auc_rf, fpr_rf, tpr_rf, feature_importances_rf
