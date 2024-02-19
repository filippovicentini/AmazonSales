import streamlit as st
from functions import *
import plotly.express as px
import warnings

#In this page I report the findings computed and I make the comparison between the two models.

filepath = '/Users/filippovicentini/Desktop/programming_project/AmazonSales/datasets/Amazon Sale Report.csv'
st.title(':blue[Findings and] :orange[Discussions] 🔍')

accuracy_test_lg, precision_test_lg, recall_test_lg, auc_lr, fpr_lr, tpr_lr, feature_importance_lr, accuracy_test_rf, precision_test_rf, recall_test_rf, auc_rf, fpr_rf, tpr_rf, feature_importance_rf = logistic_and_forest(filepath, 0.2)

# tabular presentation of performances usind pandas dataframe

data_values = {
    "Model (Classifier)":["Logistic Regression","Random Forest"],
    "Accuracy":[accuracy_test_lg,accuracy_test_rf],
    "Precision":[precision_test_lg,precision_test_rf],
    "Recall":[recall_test_lg,recall_test_rf],
    "AUC":[auc_lr,auc_rf]
}

st.subheader(':blue[Findings on Performance]', divider='orange')
performance_table = pd.DataFrame(data_values)
# visualization of the consolidated performances
fig,(ax1,ax2) = plt.subplots(1,2,figsize = (14,5))
plt.subplots_adjust(wspace = 0.5, top = 0.85)

# plot for column chart
performance_table.plot(kind = "bar",x = "Model (Classifier)",
                       y = ["Accuracy","Precision","Recall"], ax = ax1)
ax1.set_title("Accuracy, Precision & Recall", fontsize = 14)
ax1.set_ylabel("Score [%]")
ax1.set_xticks(ticks = [0,1], labels = performance_table.iloc[:,0],rotation = 0 )
ax1.set_yticks(ticks = range(20,141,20))
ax1.legend(loc = "upper right")

#data labels of "logistic regression"
x = -0.2
for y in performance_table.iloc[0,1:4]:
    ax1.text(x,y+5,str(round(y))+"%",horizontalalignment = "center")
    x = x + 0.2
    
# data labels of "random forest"
x = 0.8
for y in performance_table.iloc[1,1:4]:
    ax1.text(x,y+5,str(round(y))+"%",horizontalalignment = "center")
    x = x + 0.2


# plot the ROC curves of both the models
ax2.plot(fpr_rf,tpr_rf)
ax2.plot(fpr_lr,tpr_lr)
ax2.plot([0,1],[0,1], linestyle = "--")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curves",fontsize = 14)
ax2.fill_between(fpr_rf,tpr_rf, alpha = 0.3)
ax2.fill_between(fpr_lr,tpr_lr, alpha = 0.3)
ax2.text(0.05,0.9,f"AUC (RF)= {round(auc_rf,3)}", fontweight = 700)
ax2.text(0.05,0.5,f"AUC (LR)= {round(auc_lr,3)}", fontweight = 700)
st.pyplot(fig)

st.markdown('It is clear from the figures above that **random forest classifier** performs better than **logistic regression classifier**. A test size of 0.2 was used for the results depicted in the figure.')

st.subheader(':blue[Findings on Features]', divider='orange')

st.markdown('Also, we may conclude that the important features impacting the decision of the customer in rejecting the orders or not are **"amount"**,**"fulfillment"** and **"shipservicelevel"** as they are found having greater magnitude and in top five features, common, in both the models.')

col1, col2 = st.columns(2)

col1.caption(':blue[Logistic] :orange[Regression]')
fig, ax = plt.subplots(figsize = (7,5))
#plt.subplots_adjust(hspace = 0.5)
feature_importance_lr.tail(10).plot(kind = "barh", ax = ax)
plt.xlabel("Weights [Coefficient]",fontsize = 12)
plt.ylabel("Features", fontsize = 12)
col1.pyplot(fig)


col2.caption(':blue[Random] :orange[Forest]')
 # plot top 10 select_dtypesbottom 10 important features
fig, ax = plt.subplots(figsize = (7,5))
#plt.subplots_adjust(hspace = 0.5)
feature_importance_rf.tail(10).plot(kind = "barh", ax = ax)
plt.xlabel("Weights [Coefficient]",fontsize = 12)
plt.ylabel("Features", fontsize = 12)
col2.pyplot(fig)

st.subheader(':blue[Possible Limitations]', divider='orange')
st.markdown('We can see that 88% of the data are from orders with **single-product order**. So we dropped the **multiple-product orders** and we focused our analysis on the single-product orders for prediction/classification. Therefore, future analyses might try to take this exclusion into account')

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

# Calcola e memorizza i valori di proporzione in una Serie pandas
orderid_repeat_rows = df[df["orderid"].duplicated(keep=False)]
unique_orderid_repeat_list = orderid_repeat_rows["orderid"].unique()
plot_dataseries = round(df["orderid"].isin(unique_orderid_repeat_list).value_counts(normalize=True) * 100, 2)

# Creare una figura Streamlit
fig, ax = plt.subplots(figsize=(10,6))

# Plotta il grafico a barre
plot_dataseries.plot(kind="bar", ax=ax)
#ax.set_title("Proportions of Single-Product \nand Multipl-Product Orders", fontsize=16)
ax.set_ylabel("Proportion", fontsize=14)
ax.set_xlabel("Order Types", fontsize=14)
ax.set_xticks(plot_dataseries.index)
ax.set_xticklabels(["Single-Product Orders", "Multiple-Product Orders"], rotation="horizontal")
ax.set_yticks([20, 40, 60, 80, 100, 120])
ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%", "120%"])

# Crea altre due serie con valori da visualizzare come etichette di dati/valori nel grafico
data_label = plot_dataseries.astype(str).str.cat(np.full((2,), "%"), sep="")
count_label = pd.Series(df["orderid"].isin(unique_orderid_repeat_list).value_counts()).astype("str")

# Aggiungi/visualizza l'etichetta dei dati nel grafico
# nel formato percentuale
for x, y in enumerate(plot_dataseries):
    ax.text(x, y - 10, data_label[x], color="white",
            fontweight=500, fontsize=14, horizontalalignment="center")

# nel formato di conteggio
for x, y in enumerate(plot_dataseries):
    ax.text(x, y + 5, ("Count:\n" + count_label[x]), color="Darkgreen",
            fontweight=500, fontsize=14, horizontalalignment="center")

# Visualizza il grafico in Streamlit
st.pyplot(fig)

warnings.filterwarnings("ignore", category=FutureWarning)