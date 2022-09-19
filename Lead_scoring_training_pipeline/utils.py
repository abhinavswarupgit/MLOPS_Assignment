'''
filename: utils.py
functions: encode_features, get_train_model
creator: shashank.gupta
version: 1
'''

###############################################################################
# Import necessary modules
# ##############################################################################


import pandas as pd
import numpy as np
from sklearn import preprocessing
import sqlite3
from sqlite3 import Error

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support


###############################################################################
# Define the function to encode features
# ##############################################################################

def __read_input_data(db_path, db_file_name):
    cnx = sqlite3.connect(db_path+db_file_name)
    df = pd.read_sql('select * from interactions_mapped',cnx)
    
    print('df columns at initial read of Training Pipeline', df.columns)
    #df.drop(columns=["level_0",'index'],axis=1, inplace=True,errors="ignore")
    cnx.close()
    print("Data from the interactions_mapped")
    return df

def __save_data_to_db(db_path,db_file_name,input_data,table):
    cnx = sqlite3.connect(db_path+db_file_name)
    input_data.to_sql(name=table,con=cnx,if_exists='replace',index=False)
    print("DataBase Table Created : ", table)
    cnx.close()
    

def encode_features(db_path, db_file_name, one_hot_encoded_features, features_to_encode):
    '''
    This function one hot encodes the categorical features present in our  
    training dataset. This encoding is needed for feeding categorical data 
    to many scikit-learn models.

    INPUTS
        db_file_name : Name of the database file 
        db_path : path where the db file should be
        ONE_HOT_ENCODED_FEATURES : list of the features that needs to be there in the final encoded dataframe
        FEATURES_TO_ENCODE: list of features  from cleaned data that need to be one-hot encoded
       

    OUTPUT
        1. Save the encoded features in a table - features
        2. Save the target variable in a separate table - target


    SAMPLE USAGE
        encode_features()
        
    **NOTE : You can modify the encode_featues function used in heart disease's inference
        pipeline from the pre-requisite module for this.
    '''
    input_data = __read_input_data(db_path,db_file_name)
    
    #first we will drop the created_date columns
    input_data.drop(['created_date'],axis=1, inplace=True)
    print('df columns after created_date Drop', input_data.columns)
    
    print('features_to_encode: ', features_to_encode)
    print(input_data[features_to_encode].head())
    print('input_data shape: ', input_data.shape)
    
    #encoded_df = input_data[features_to_encode]
    #one hot encoding
    encoded_df = pd.get_dummies(data=input_data, columns=features_to_encode)
    print('After One Hot Encoding columns: ' , encoded_df.columns)
    print(encoded_df.head())
    
    #selection based on one_hot_encoded_features constant value
    print('one_hot_encoded_features columns: ',one_hot_encoded_features)
    encoded_df = encoded_df[one_hot_encoded_features]
    print('After Feature Selections: ' , encoded_df.columns)
    print('After Feature Selections: data type ' , encoded_df.dtypes)
    print('After Feature Selections: shape ' , encoded_df.shape)
    print('After Feature Selections: Null Values ' , encoded_df.isnull().sum())
    
    #get X & y for model train
    X = encoded_df.drop('app_complete_flag',axis=1)
    y = encoded_df[['app_complete_flag']]
    
    #write to database table
    __save_data_to_db(db_path, db_file_name, X, 'features')
    __save_data_to_db(db_path, db_file_name, y, 'target')
    
    print('Feature & target Table created in DB : Success')
###############################################################################
# Define the function to train the model
# ##############################################################################

def get_trained_model(db_path, db_file_name, model_config, experiment, tracking_uri):
    '''
    This function setups mlflow experiment to track the run of the training pipeline. It 
    also trains the model based on the features created in the previous function and 
    logs the train model into mlflow model registry for prediction. The input dataset is split
    into train and test data and the auc score calculated on the test data and
    recorded as a metric in mlflow run.   

    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be


    OUTPUT
        Tracks the run in experiment named 'Lead_Scoring_Training_Pipeline'
        Logs the trained model into mlflow model registry with name 'LightGBM'
        Logs the metrics and parameters into mlflow run
        Calculate auc from the test data and log into mlflow run  

    SAMPLE USAGE
        get_trained_model()
    '''
    cnx = sqlite3.connect(db_path+db_file_name)
    
    #read target & featuers
    X = pd.read_sql('select * from features',cnx)
    print('Feature Variable', X.columns)
    
    y = pd.read_sql('select * from target',cnx)
    print('Target Variable', y.columns)
    
    X.drop(columns=['index'],axis=1,inplace=True,errors='ignore')
    y.drop(columns=['index'],axis=1,inplace=True,errors='ignore')
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
    
    mlflow.set_tracking_uri(tracking_uri)
    
    with mlflow.start_run(run_name=experiment) as mlrun:
        clf = lgb.LGBMClassifier()
        clf.set_params(**model_config) # add ** airflow throws an error
        clf.fit(X_train, y_train)

        mlflow.sklearn.log_model(sk_model=clf, artifact_path="models",  registered_model_name='LightGBM')
        mlflow.log_params(model_config)    

        # predict the results on training dataset
        y_pred=clf.predict(X_test)

        # view accuracy
        acc=accuracy_score(y_pred, y_test)
        conf_mat = confusion_matrix(y_pred, y_test)
       
        precision = precision_score(y_pred, y_test,average= 'macro')
        recall = recall_score(y_pred, y_test, average= 'macro')
        cm = confusion_matrix(y_test, y_pred)
        tn = cm[0][0]
        fn = cm[1][0]
        tp = cm[1][1]
        fp = cm[0][1]
        class_zero = precision_recall_fscore_support(y_test, y_pred, average='binary',pos_label=0)
        class_one = precision_recall_fscore_support(y_test, y_pred, average='binary',pos_label=1)

        mlflow.log_metric('test_accuracy', acc)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("Precision_0", class_zero[0])
        mlflow.log_metric("Precision_1", class_one[0])
        mlflow.log_metric("Recall_0", class_zero[1])
        mlflow.log_metric("Recall_1", class_one[1])
        mlflow.log_metric("f1_0", class_zero[2])
        mlflow.log_metric("f1_1", class_one[2])
        mlflow.log_metric("False Negative", fn)
        mlflow.log_metric("True Negative", tn)
        
        print('Trained Model : Success')