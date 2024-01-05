#libraries
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns

#this function shows with a graphic the distribution of NaN values in the dataset
def where_nan(df):
    sns.heatmap(df.isnull())
    plt.title("Distribution of NaN Values")
    plt.show()

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