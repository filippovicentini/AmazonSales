#libraries
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
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

#this function shows with a graphic the distribution of NaN values in the dataset
def where_nan(df):
    sns.heatmap(df.isnull())
    plt.title("Distribution of NaN Values")
    plt.show()

#this function show the net revenue
def amazon_net_revenue(df):
    sns.set_style('whitegrid')

    # Group the data by month and calculate the total sales revenue
    monthly_sales = df.groupby(pd.Grouper(key='date', freq='M')).agg({'order_amount_($)': 'sum'})

    # Get latest month revenue and average quarterly revenue
    latest_month_revenue = monthly_sales.iloc[-1, 0]
    avg_quarterly_revenue = monthly_sales.iloc[-3:-1, 0].mean()

    # Compute percentage below average revenue for quarter
    pct_below_avg = round((1 - (latest_month_revenue / avg_quarterly_revenue)) * 100, 1)

    # Plot the monthly sales revenue
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(monthly_sales.index.strftime('%b'), monthly_sales['order_amount_($)'], color='#878787')

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
    plt.axhline(avg_quarterly_revenue, linestyle='--', color='orange',linewidth=2, label='Q2 Average Revenue')

    ax.set_title('Amazon India Net Revenue', fontsize=20, x=.19, y=1.05)
    ax.text(-.08, 1.02, 'Q2 FY22', fontsize=15, color='#878787', transform=ax.transAxes)
    ax.set_xlabel(None)
    ax.set_yticks(list(range(0, 41, 5)))
    ax.set_yticklabels(list(range(0, 41, 5)))

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
    plt.show()

#this function shows the average monthly order amount
def average_monthly_order_amount(df):
    # Group the data by month and calculate the average order value
    monthly_aov = df.groupby(pd.Grouper(key='date', freq='M')).agg({'order_amount_($)': 'sum', 'order_ID': 'nunique'})
    monthly_aov['average_order_value'] = monthly_aov['order_amount_($)'] / monthly_aov['order_ID']

    # Calculate percent change from previous month
    monthly_aov['pct_change'] = monthly_aov['average_order_value'].pct_change() * 100

    # Create a barplot of the average order value per month
    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(x=monthly_aov.index.strftime('%b'), y=monthly_aov['average_order_value'], ax=ax, color='#878787')

    # Add line plot of the average order value per month
    ax.plot(monthly_aov.index.strftime('%b'), monthly_aov['average_order_value'], linestyle='--', linewidth=2, color='orange', marker='o')


    # Add callout for percent increase from April to June
    apr_val = monthly_aov['average_order_value'].iloc[0]
    jun_val = monthly_aov['average_order_value'].iloc[2]
    pct_change = ((jun_val - apr_val) / apr_val) * 100
    ax.annotate(f'Increase of {pct_change:.2f}% from Apr to Jun',fontweight='bold', xy=(2,8.074941567466606), xytext=(1.65, 8.264941567466606), fontsize=13, ha='center', va='bottom', arrowprops=dict(arrowstyle='->', color='black', lw=1.5, connectionstyle="arc3,rad=-0.1"))

    # Set labels and title
    ax.set_title('Average Monthly Order Amount', fontsize=20, x=.22, y=1.07)
    ax.text(-0.09, 1.04, 'Q2 FY22', fontsize=15, color='#878787', transform=ax.transAxes)
    ax.set_xlabel(None)
    ax.set_ylabel('Average Order Value ($)', fontsize=12, labelpad=3)
    ax.set_ylim(7.20, 8.50)
    ax.yaxis.grid(linestyle='--', color='gray', linewidth=0.5, dashes=(8, 5))

    ax.tick_params(axis='both', labelsize=12)
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_color('black')

    plt.show()

#this function shows the top product revenue by month
def top_product_revenue_by_month(df):
    import warnings
    warnings.filterwarnings('ignore')

    fig, ax = plt.subplots(figsize=(8,6))

    # Define the desired order of months
    month_order = ['April', 'May', 'June']

    # Filter the data to only include the four product categories of interest
    sales_data = df[df['product_category'].isin(['Western Dress', 'Top', 'kurta', 'Set'])]

    # Convert the date column to a datetime object
    sales_data['date'] = pd.to_datetime(sales_data['date'])

    # Extract the month from the date column and set it as a new column
    sales_data['month'] = sales_data['date'].dt.month_name()

    # Aggregate the sales data by month and product category
    sales_by_month = sales_data.groupby(['month', 'product_category'])['order_amount_($)'].sum().reset_index()

    # Convert the month column to a categorical data type with the desired order
    sales_by_month['month'] = pd.Categorical(sales_by_month['month'], categories=month_order, ordered=True)

    # Plot the sales data using seaborn
    ax = sns.barplot(x='month', y='order_amount_($)', hue='product_category', data=sales_by_month,
                    palette=['#969696', '#bdbdbd', 'orange', '#d9d9d9'])

    # Extract the sales data for Western Dress
    sales_wd = sales_by_month[sales_by_month['product_category'] == 'Western Dress'].reset_index(drop=True)
    sales_wd['month'] = pd.Categorical(sales_wd['month'], categories=month_order, ordered=True)
    sales_wd.sort_values(by='month',inplace=True)
    # Add line plot for total monthly revenue of Western Dress
    ax.plot([0.1,1.1,2.1], sales_wd['order_amount_($)'], color='black', linestyle='--', linewidth=2, marker='o')


    # Add annotation for percent increase from April to June for Western Dress
    pct_increase = (sales_wd.loc[1, 'order_amount_($)'] - sales_wd.loc[0, 'order_amount_($)']) / sales_wd.loc[0, 'order_amount_($)'] * 100
    ax.annotate(f'{pct_increase:.0f}% increase\n April to June',fontweight='bold', xy=(2.1, sales_wd.loc[2, 'order_amount_($)']), xytext=(1.88, sales_wd.loc[2, 'order_amount_($)'] + 40000),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5, connectionstyle="arc3,rad=0.1"))


    # Set the number of y ticks you want
    num_y_ticks = 10

    # Calculate the y tick values
    y_tick_values = np.linspace(ax.get_yticks()[0], ax.get_yticks()[-1], num_y_ticks)

    # Set the y ticks
    ax.set_yticks(y_tick_values)


    # Add title and axis labels
    ax.set_title('Top Product Revenue by Month', fontsize=20, x=.22, y=1.07)
    ax.text(-0.09, 1.04, 'Q2 FY22', fontsize=15, color='#878787', transform=ax.transAxes)

    plt.legend(bbox_to_anchor=(1,1), fontsize=12, framealpha=1)

    ax.set_xlabel(None)
    ax.set_ylabel('Net Revenue in 10,000 dollars', fontsize=12, labelpad=3)
    ax.set_yticklabels(list(range(0,46,5)))
    ax.yaxis.grid(linestyle='--', color='gray', linewidth=0.5, dashes=(8, 5))

    ax.tick_params(axis='both', labelsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_color('black')


    # Show the plot
    plt.show()
    warnings.filterwarnings('default')  # Re-enable the warnings

#this function shows the sales by product size
def sales_by_product_size(df):
    # Group the data by product size and calculate the total sales
    sales_by_size = df.groupby('size')['order_amount_($)'].sum()

    # Create a horizontal bar chart to show the sales by product size
    fig, ax = plt.subplots(figsize=(12,6))

    # Use a color palette to highlight specific sizes
    palette_colors = ['orange' if size in ['S', 'M', 'L'] else '#878787' for size in sales_by_size.index]
    sns.barplot(x=sales_by_size.index, y=sales_by_size.values, ax=ax, palette=palette_colors)


    # Set font sizes for x and y labels, title, and ticks
    ax.set_xlabel('Product Size', labelpad=3, fontsize=14)
    ax.set_ylabel('Net Revenue in 10,000 dollars', labelpad=3, fontsize=14)
    ax.set_yticklabels(list(range(0,20,2)))
    ax.set_title('Sales by Product Size', fontsize=20, x=0.085, y=1.05, pad=10)
    ax.text(-0.06, 1.04, 'Q2 FY22', fontsize=15, color='#878787', transform=ax.transAxes)
    #ax.set_title('Top Product Revenue by Month', fontsize=20, x=.22, y=1.07)


    ax.tick_params(axis='both', labelsize=12)
    ax.yaxis.grid(linestyle='--', color='gray', linewidth=0.5, dashes=(8, 5))
    ax.xaxis.grid(False)


    # Set the number of y ticks you want
    num_y_ticks = 10

    # Calculate the y tick values
    y_tick_values = np.linspace(ax.get_yticks()[0], ax.get_yticks()[-1], num_y_ticks)

    # Set the y ticks
    ax.set_yticks(y_tick_values)

    # Set font sizes for the bars and add annotations for S, M, and L sizes
    for i, size in enumerate(sales_by_size.index):
        if size in ['S', 'M', 'L']:
            ax.text(i, sales_by_size.values[i], f'{sales_by_size.values[i]/10000:.0f}k', ha='center', fontsize=14, fontweight='bold', color='black')


    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_color('black')

    plt.show()

#heatmap of quantity sold by category and size
def heatmap_category_size(df):
    heatmap_data = df.pivot_table(index='product_category', columns='size', values='order_quantity', 
                                  aggfunc='sum', fill_value=0)
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='d', linewidths=.5)
    plt.title('Heatmap of Quantity Sold by Category and Size')
    plt.show()

#top 10 cities with the most orders
def top_cities(df):
    pd.value_counts(df['city'])[0:10].plot(kind = 'pie' , autopct = '%1.0f%%')
    plt.title('Top 10 cities with the most orders' , fontsize = 16, fontweight = "bold")
    plt.show()

#this function cleans the original dataset
def df_cleaning(df):
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
    mapper = {'Order ID':'order_ID', 'Date':'date', 'Status':'ship_status','Fulfilment':'fullfilment',
          'ship-service-level':'service_level', 'Style':'style', 'SKU':'sku', 'Category':'product_category', 
          'Size':'size', 'ASIN':'asin', 'Courier Status':'courier_ship_status', 'Qty':'order_quantity', 
          'Amount':'order_amount_($)', 'ship-city':'city', 'ship-state':'state', 'ship-postal-code':'zip', 
          'promotion-ids':'promotion','B2B':'customer_type'}
    df.rename(columns=mapper, inplace =True)
    exchange_rate = 0.0120988
    df['order_amount_($)'] = df['order_amount_($)'].apply(lambda x: x * exchange_rate)
    df['customer_type'].replace(to_replace=[True,False],value=['business','customer'], inplace=True)
    df['date'] = pd.to_datetime(df['date'], format='%m-%d-%y')
    march_dates = df['date'][df['date'].dt.month == 3]
    df = df[(df['date'].dt.month != 3)]
    df['month'] = df['date'].dt.month
    month_map = { 4: 'april',5: 'may',6: 'june'}
    df['month'] = df['date'].dt.month.map(month_map)
    month_order = ['april', 'may', 'june']
    df['month'] = pd.Categorical(df['month'], categories=month_order, ordered=True)
    size_order = ['Free','XS', 'S', 'M', 'L', 'XL', 'XXL', '3XL', '4XL', '5XL', '6XL']
    df['size'] = pd.Categorical(df['size'], categories=size_order, ordered=True)

#this function predicts the likelihood of order rejection
def order_rejection(df):
    df_rejection = df.copy()
    # bar chart using matplotlib package
    fig,ax = plt.subplots(figsize = (7,5))
    # calculate and store the proportion values in a pandas.Series
    orderid_repeat_rows = df_rejection[df_rejection["order_ID"].duplicated(keep = False)]
    unique_orderid_repeat_list = orderid_repeat_rows["order_ID"].unique()
    plot_dataseries = round(df_rejection["order_ID"].isin(unique_orderid_repeat_list).
                        value_counts(normalize = True)*100,2)

    # plot the bar chart
    plot_dataseries.plot(kind = "bar",ax =ax)
    plt.title("Proportions of Single-Product \nand Multipl-Product Orders", fontsize = 16)
    plt.ylabel("Proportion", fontsize = 14)
    plt.xlabel("Order Types", fontsize = 14)
    plt.xticks(ticks = plot_dataseries.index,
            labels = ["Single-Product Orders", "Mutiple-Product Orders"], rotation = "horizontal")
    plt.yticks(ticks = [20,40,60,80,100,120], labels = ["20%","40%","60%","80%","100%","120%"])

    # create another two series with values to be diaplayed as data-label/value-label in the chart
    data_label = plot_dataseries.astype(str).str.cat(np.full((2,),"%"), sep = "")
    count_label = pd.Series(df_rejection["order_ID"].isin(unique_orderid_repeat_list).value_counts()).astype("str")

    # add/plot the data-label in the chart
    # in percentage format
    for x,y in enumerate(plot_dataseries):
        plt.text(x,y-10,data_label[x],color = "white",
                fontweight = 700,fontsize = 14, horizontalalignment = "center")
    
    # in count values format
    for x,y in enumerate(plot_dataseries):
        plt.text(x,y+5,("Count:\n"+count_label[x]),color = "Darkgreen",
                fontweight = 700,fontsize = 14, horizontalalignment = "center")
    plt.show()
    # drop the orderids with multiple products
    df_rejection = df_rejection[df_rejection["order_ID"].duplicated(keep = False) == False]
    # drop redundant column "orderid"
    df_rejection.drop("order_ID", axis = 1, inplace = True)
    #Create a column named "rejected" as the target feature with two unique values (classifications): 1 and 0 representing rejected and not-rejected respectively
    # drop the rows with unsure rejection status
    known_value = ["Cancelled", 'Shipped - Returned to Seller','Shipped - Rejected by Buyer',
                'Shipped - Returning to Seller','Shipped - Delivered to Buyer']
    df_rejection = df_rejection[df_rejection['ship_status'].isin(known_value)]   

    # create a col "rejected" where value 1 means rejected and 0 means not-rejected" 
    rejected = ["Cancelled", 'Shipped - Returned to Seller','Shipped - Rejected by Buyer',
                'Shipped - Returning to Seller']
    df_rejection["rejected"] = df['ship_status'].isin(rejected).astype(int)    # change the dtype to "int" 

    # drop col "status" 
    df_rejection.drop("ship_status",axis = "columns", inplace = True)
    # bar chart using matplotlib package
    fig,ax = plt.subplots(figsize = (7,5))

    # calculate and store the proportion values in a pandas.Series
    plot_dataseries = round(df_rejection["rejected"].value_counts(normalize = True)*100,2)

    # plot the bar chart
    plot_dataseries.plot(kind = "bar",ax =ax)
    plt.title("Proportions of rejected and not-rejected orders", fontsize = 14)
    plt.ylabel("Proportion", fontsize = 12)
    plt.xlabel("Status", fontsize = 12)
    plt.xticks(ticks = range(len(plot_dataseries)),
            labels = ["Not Rejected", "Rejected"], rotation = "horizontal")
    plt.yticks(ticks = [20,40,60,80,100], labels = ["20%","40%","60%","80%","100%"])

    # create another series with values to be diaplayed as data-label/value-label in the chart
    data_label = plot_dataseries.astype(str).str.cat(np.full((2,),"%"), sep = "")

    # create one more series to to display count
    data_count = df_rejection["rejected"].value_counts()

    # add/plot the data-label in the chart
    for x,y in enumerate(plot_dataseries):
        plt.text(x,y-10,data_label[x],color = "white",
                fontweight = 700,fontsize = 13, horizontalalignment = "center")

    # add count label
    for x,y in enumerate(plot_dataseries):
        plt.text(x,y+5,"Count:\n" + str(data_count[x]),fontweight = 700,
                fontsize = 13,horizontalalignment = "center")
    
    plt.show()
    print(df_rejection.info())
    # drop the redundant cols " in the dataset
    df_rejection.drop(["date","fullfilment","promotion","courier_ship_status"],axis ="columns", inplace = True)
    # drop high cardinality features
    df_rejection.drop(["style","sku","city","zip","asin"],axis = 1, inplace = True)
    df_rejection.drop("order_quantity", axis = "columns", inplace = True)
    df_rejection = df_rejection[(df_rejection["order_amount_($)"] < df_rejection["order_amount_($)"].quantile(0.95)) | df["order_amount_($)"].isnull()]
    df_rejection["region"] = df_rejection["state"].replace({
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
    df_rejection = df_rejection[df_rejection["state"] != "unknown"]
    df_rejection.drop("state",axis = "columns", inplace = True)

    #Split the Dataset into target vector and feature matrix
    target = "rejected"
    y = df_rejection[target]     # target vector
    X = df_rejection.drop(target, axis = "columns")   # feature matrix/dataframe

    #display target vector and feature matrix
    print("Target Vector:")
    print(y.head())
    print("\nThe feature matrix:")
    print(X.head())
    
    # split the datasets into training and test datasets
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size= 0.2,random_state = 42)

    # print the shape of the training and testing datasets
    print(f"Shape of X_train :{X_train.shape}\nShape of X_test: {X_test.shape}",
        f"\nShape of y_train: {y_train.shape}\nShape of y_test: {y_test.shape}")