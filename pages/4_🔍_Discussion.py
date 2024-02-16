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

st.markdown('It is clear from the figures above that **random forest classifier** performs better than **logistic regression classifier**.')

st.subheader(':blue[Findings on Features]', divider='orange')

st.markdown('Also, we may conclude that the important features impacting the decision of the customer in rejecting the orders or not are **"amount"**,**"merchant"** and **"standard"** as they are found having greater magnitude and in top five features, common, in both the models.')

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

warnings.filterwarnings("ignore", category=FutureWarning)