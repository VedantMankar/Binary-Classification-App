import streamlit as st 
import pandas as pd 
import numpy as np 
import os 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix,plot_roc_curve,plot_precision_recall_curve
from sklearn.metrics import precision_score,accuracy_score,recall_score

x = None 
y = None
tar_col = None
label_list = []
load = None
def main():
    st.title("Machine Learning")
    st.subheader("Machine Learning app for Binary classification")

    st.markdown("Streamlit is reaaly cool")
    def load_data(file_path):
        global load
        if file_path:
            try:
                data = pd.read_csv(file_path)
                st.write("Data Loaded succesfully")
                load = True
                return data 
            except FileNotFoundError :
                st.write("Please enter correct file name")
                load = False
    def file_selector(folder_path ='./datasets'):
        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox("Select a File",filenames)
        return os.path.join(folder_path,selected_filename)

    def data_shape(df):
        return df.shape

    def encode_data(df):

        label = LabelEncoder()
        for col in df.columns:
            df[col] = label.fit_transform(df[col])
        return df
      
    def select_target_cols(df):
        global tar_col
        tar_col = st.text_input("Enter column to be predicted!.")
        #st.help("Enter the name of column which we want to predict")
        
        if tar_col in df.columns:
            st.text("Column {} found".format(tar_col))
        else:
            st.text("Enter correct columns!!")
    
    def split(df,random_state,test_size):
        try:
            global x , y 
            y = df[tar_col].values
            x = df.drop(columns=tar_col).values
            x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=random_state,test_size=test_size)
            return x_train,x_test,y_train,y_test
        except KeyError:
            st.write("Please Select Dependent(X) and Independent Columns(Y) ")
     
    def select_label(df,tar_col):
        try:
            label = df[tar_col].unique()
            
            for i in range(len(label)):
                label_list.append(st.sidebar.text_input("Enter Name for label {} ".format(i+1)))
            return label_list
        except KeyError:
            st.sidebar.text("Please select the Output column by clicking on <b>select_target_cols<b>. ")
        
    def plot_metrics(metrics_list,model):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model,x_test,y_test,display_labels=label_list)
            st.pyplot()
        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model,x_test,y_test)
            st.pyplot()
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model,x_test,y_test)
            st.pyplot()


    def train_model(x_train,x_test,y_train,y_test,classifier):
        if classifier == 'Support Vector Classifier(SVC)':
            st.sidebar.subheader("Model Hyperparameters")
            C = st.sidebar.number_input("C (Regularization parameter)",0.01,10.0,step=0.01,key='C')
            kernel = st.sidebar.radio("Kernel",("rbf","linear"),key="kernel")
            gamma = st.sidebar.radio("Gamma (Kernel Coefficeint)",("scale","auto"),key="gamma")
            metrics = st.sidebar.multiselect("What metric to plot?",("Confusion Matrix","ROC Curve","Precision-Recall Curve"))

            if st.sidebar.button("Classify",key="classify"):
                try:
                    st.subheader("Support Vector Machine(SVM) Results")
                    model = SVC(C=C,kernel=kernel,gamma=gamma)
                    model.fit(x_train,y_train)
                    accuracy = model.score(x_test,y_test)
                    y_pred = model.predict(x_test)
                    st.write("Accuracy: ",accuracy.round(2))
                    st.write("Precision: ",precision_score(y_test,y_pred,labels=label_list).round(2))
                    st.write("Recall: ",recall_score(y_test,y_pred,labels=label_list).round(2))
                    plot_metrics(metrics,model)
                    
                except ValueError:
                    st.write("Could not convert string to float.Please use LabelEncoding to encode the data ")
        if classifier == 'Logistic Regression':
            st.sidebar.subheader("Model Hyperparameters")
            C = st.sidebar.number_input("C (Regularization parameter)",0.01,10.0,step=0.01,key='C_Lr')
            max_iter = st.sidebar.slider("Maximum Number of Iterations: ",100,500,key='max_iter')
            solver = st.sidebar.radio("Solver",("newton-cg","lbfgs","liblinear","sag","saga"))
            metrics = st.sidebar.multiselect("What metric to plot?",("Confusion Matrix","ROC Curve","Precision-Recall Curve"))

            if st.sidebar.button("Classify",key="classify"):
                try:
                    st.subheader("Logistic Regression")
                    model = LogisticRegression(C=C,max_iter=max_iter,solver=solver)
                    model.fit(x_train,y_train)
                    accuracy = model.score(x_test,y_test)
                    y_pred = model.predict(x_test)
                    st.write("Accuracy: ",accuracy.round(2))
                    st.write("Precision: ",precision_score(y_test,y_pred,labels=label_list).round(2))
                    st.write("Recall: ",recall_score(y_test,y_pred,labels=label_list).round(2))
                    plot_metrics(metrics,model)
                except ValueError:
                    st.write("Could not convert string to float .Please use Label Encoding ")

        if classifier == 'Random Forest':
            st.sidebar.subheader("Model Hyperparameters")
            n_estimator = st.sidebar.number_input("Number of Trees ",100,500,step=10,key='n_esimator')
            max_depth = st.sidebar.number_input("The maximum depth of tree",1,20,step=1,key='max_depth')
            criterion = st.sidebar.radio("Criterion",("gini","entropy"))
            bootstrap = st.sidebar.radio("Bootstrap samples when building trees",("True","False"),key='bootstrap')
            metrics = st.sidebar.multiselect("What metric to plot?",("Confusion Matrix","ROC Curve","Precision-Recall Curve"))

            if st.sidebar.button("Classify",key="classify"):
                try:
                    st.subheader("Random Forest")
                    model = RandomForestClassifier(n_estimators=n_estimator,criterion=criterion,max_depth=max_depth,bootstrap=bootstrap)
                    model.fit(x_train,y_train)
                    accuracy = model.score(x_test,y_test)
                    y_pred = model.predict(x_test)
                    st.write("Accuracy: ",accuracy.round(2))
                    st.write("Precision: ",precision_score(y_test,y_pred,labels=label_list).round(2))
                    st.write("Recall: ",recall_score(y_test,y_pred,labels=label_list).round(2))
                    plot_metrics(metrics,model)
                except ValueError:
                    st.write("Could not convert string to float .Please use Label Encoding")

    filename = file_selector()
    df = load_data(filename)
    classifier = st.sidebar.selectbox("Classify",("Support Vector Classifier(SVC)","Logistic Regression","Random Forest"))
    

    if load:
        if st.sidebar.checkbox("Display data",False):
            st.subheader("Data")
            st.write(df)
        if st.sidebar.checkbox("Data shape",False):
            data_s = data_shape(df)
            st.write(data_s)
        if  st.sidebar.checkbox("Label Encoding",False):
            data_l = encode_data(df)
            st.write(data_l)
        if st.checkbox("Select Variable to be predicted(Y)",False):
            select_target_cols(df)
        
        if st.checkbox("Split data",False):
            try:
                random_state = st.slider("Random State",0,100,key='random_state')
                test_size = st.number_input("Test Size",0.20,0.50,step=0.01,key='test_size')
                x_train,x_test,y_train,y_test = split(df,random_state,test_size)
                st.text("Data Splitted Succesfully")
            except TypeError:
                st.write("!!!!!!!")
        if st.sidebar.checkbox("Give labels",False):
            label = select_label(df,tar_col)
        if st.sidebar.checkbox("Train Model",False):
            train_model(x_train,x_test,y_train,y_test,classifier)





if __name__ == "__main__":
    main()