import streamlit as st
import pandas as pd
import numpy as np
from functions import *
import warnings

#In this page I use Linear Regression and Random Forest Classifier in order to try to
#predict order rejection and to understand the major features that affect it.

filepath = '/Users/filippovicentini/Desktop/programming_project/AmazonSales/datasets/Amazon Sale Report.csv'
df = df_cleaning_modeling_phase(filepath)
df['amount'].fillna(0,inplace=True)

st.title(':blue[Classification Models] :orange[for Prediction]')

st.markdown("""
We shall make **ORDER REJECTION CLASSIFICATION PREDICTION** using two models based on:
  - **Logistic Regression Classifier**
  - **Random Forest Classifier**

In addition, we would like to try to find out the features that most influence this event.
""")

#DATA POCESS

st.subheader(':blue[Data] :orange[Preprocessing]', divider = 'orange')

target = "rejected"
y = df[target]     # target vector
X = df.drop(target, axis = "columns")   # feature matrix/dataframe
size_test = st.slider('Select the size of the test', min_value=0.0, max_value=1.0, value=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size_test, random_state = 42)
# resample it (training dataset) using Random Over Sampling
ros = RandomOverSampler(random_state = 42)
X_train_over,y_train_over = ros.fit_resample(X_train, y_train)

st.markdown("""In this dataset we have to address  imbalance in the target class distribution in the training-target vector. So we have to resample the training datasets to obtain balanced target class distribution.""")
show_imbalance = st.toggle('Show target class distribution before resampling')
if show_imbalance:
    plot_class_distr(y_train)

show_balance = st.toggle("Show target class distribution after resampling")
if show_balance:
    plot_class_distr(y_train_over)
    st.sidebar.info("For predictions we will use the resample dataset!")

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
X_trainrf, X_testrf, y_trainrf, y_testrf = train_test_split(X_encoded, y, test_size=size_test, random_state=42)
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

#GAUSSIAN
#gaus = GaussianNB()
#gaus.fit(X_trainrf, y_trainrf)
#y_pred_gaus = model.predict(X_testrf)
#conf_matrix_test_gaus = confusion_matrix(y_testrf, y_pred_gaus)
#accuracy_test_gaus = round(gaus.score(X_testrf,y_testrf)*100,2)
#precision_test_gaus = round(conf_matrix_test_gaus[1,1]/(conf_matrix_test_gaus[1,1]+conf_matrix_test_gaus[0,1])*100,2)
#recall_test_gaus = round(conf_matrix_test_gaus[1,1]/(conf_matrix_test_gaus[1,1]+conf_matrix_test_gaus[1,0])*100,2)
#fig,ax = plt.subplots(figsize = (5,5))
#ConfusionMatrixDisplay.from_estimator(gaus,X_testrf,y_testrf,
                                        #ax = ax, colorbar=False)
#st.pyplot(fig)
#st.sidebar.info(f"Model Evaluation Metrics:\n"
                #f"  - Accuracy Score: {accuracy_test_gaus}%\n"
                #f"  - Precision Score: {precision_test_gaus}%\n"
                #f"  - Recall Score: {recall_test_gaus}%")
#VISUALIZATION

st.subheader(':blue[Process] :orange[Data]', divider='orange')

choose_model = st.radio('Select the Model',['Logistic Regression','Random Forest'],
                        index=None, horizontal=True)

choose_graph = st.radio('Choose the graph', ['Confusion Matrix Training', 'Confusion Matrix Test','ROC Curve', 'Top 10 Features'],
                        index=None,horizontal=True)

if choose_model == 'Logistic Regression':
    if choose_graph == 'Confusion Matrix Training':
        fig,ax = plt.subplots(figsize = (5,5))
        ConfusionMatrixDisplay.from_estimator(logistic_model,X_train_over,y_train_over,
                                        ax = ax, colorbar=False)
        st.pyplot(fig)
        st.sidebar.metric(label='Accuracy Score', value=accuracy_train_lg)
        st.sidebar.metric(label='Precision Score', value=precision_train_lg)
        st.sidebar.metric(label='Recall Score', value=recall_train_lg)
    if choose_graph == 'Confusion Matrix Test':
        fig,ax = plt.subplots(figsize = (5,5))
        ConfusionMatrixDisplay.from_estimator(logistic_model,X_test,y_test,
                                        ax = ax, colorbar=False)
        st.pyplot(fig)
        st.sidebar.metric(label='Accuracy Score', value=accuracy_test_lg)
        st.sidebar.metric(label='Precision Score', value=precision_test_lg)
        st.sidebar.metric(label='Recall Score', value=recall_test_lg)
    if choose_graph == 'ROC Curve':
        data = pd.DataFrame({
        'False Positive Rate': fpr_lr,
        'True Positive Rate': tpr_lr
        })

        # Crea un grafico ROC Curve con area sottesa evidenziata
        chart = alt.Chart(data).mark_area(opacity=0.3, interpolate='step').encode(
            x='False Positive Rate',
            y='True Positive Rate'
        )

        # Aggiungi la curva ROC al grafico
        line = alt.Chart(data).mark_line(color='#87ceeb').encode(
            x='False Positive Rate',
            y='True Positive Rate'
        )

        # Aggiungi la retta y=x al grafico
        baseline = alt.Chart(pd.DataFrame({'x': [0, 1], 'y': [0, 1]})).mark_line(color='red', strokeDash=[5, 5]).encode(
            x='x',
            y='y'
        )

        chart = chart + line + baseline

        st.altair_chart(chart, use_container_width=True)
        st.sidebar.metric(label='AUC', value=round(auc_lr,3))
    if choose_graph == 'Top 10 Features':
        # plot top 10 select_dtypesbottom 10 important features
        fig, ax = plt.subplots(figsize = (7,5))
        #plt.subplots_adjust(hspace = 0.5)
        feature_importances_lr.tail(10).plot(kind = "barh", ax = ax)
        plt.xlabel("Weights [Coefficient]",fontsize = 12)
        plt.ylabel("Features", fontsize = 12)
        st.pyplot(fig)


if choose_model == 'Random Forest':
    if choose_graph == 'Confusion Matrix Training':
        fig,ax = plt.subplots(figsize = (5,5))
        ConfusionMatrixDisplay.from_estimator(model,X_trainrf,y_trainrf,
                                        ax = ax, colorbar=False)
        st.pyplot(fig)
        st.sidebar.metric(label='Accuracy Score', value=accuracy_train_rf)
        st.sidebar.metric(label='Precision Score', value=precision_train_rf)
        st.sidebar.metric(label='Recall Score', value=recall_train_rf)
    if choose_graph == 'Confusion Matrix Test':
        fig,ax = plt.subplots(figsize = (5,5))
        ConfusionMatrixDisplay.from_estimator(model,X_testrf,y_testrf,
                                        ax = ax, colorbar=False)
        st.pyplot(fig)
        st.sidebar.metric(label='Accuracy Score', value=accuracy_test_rf)
        st.sidebar.metric(label='Precision Score', value=precision_test_rf)
        st.sidebar.metric(label='Recall Score', value=recall_test_rf)
    if choose_graph == 'ROC Curve':
        data = pd.DataFrame({
        'False Positive Rate': fpr_rf,
        'True Positive Rate': tpr_rf
        })

        # Crea un grafico ROC Curve con area sottesa evidenziata
        chart = alt.Chart(data).mark_area(opacity=0.3, interpolate='step').encode(
            x='False Positive Rate',
            y='True Positive Rate'
        )

        # Aggiungi la curva ROC al grafico
        line = alt.Chart(data).mark_line(color='#87ceeb').encode(
            x='False Positive Rate',
            y='True Positive Rate'
        )

        # Aggiungi la retta y=x al grafico
        baseline = alt.Chart(pd.DataFrame({'x': [0, 1], 'y': [0, 1]})).mark_line(color='red', strokeDash=[5, 5]).encode(
            x='x',
            y='y'
        )

        chart = chart + line + baseline

        st.altair_chart(chart, use_container_width=True)
        st.sidebar.metric(label='AUC', value=round(auc_rf,3))

    if choose_graph == 'Top 10 Features':
        # plot top 10 select_dtypesbottom 10 important features
        fig, ax = plt.subplots(figsize = (7,5))
        #plt.subplots_adjust(hspace = 0.5)
        feature_importances_rf.tail(10).plot(kind = "barh", ax = ax)
        plt.xlabel("Weights [Coefficient]",fontsize = 12)
        plt.ylabel("Features", fontsize = 12)
        st.pyplot(fig)
    


warnings.filterwarnings("ignore", category=FutureWarning)













