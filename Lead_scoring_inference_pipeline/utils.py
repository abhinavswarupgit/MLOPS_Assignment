'''
filename: utils.py
functions: encode_features, load_model
creator: shashank.gupta
version: 1
'''

###############################################################################
# Import necessary modules
# ##############################################################################

import mlflow
import mlflow.sklearn
import pandas as pd
import collections

import sqlite3

import os
import logging

from datetime import datetime

###############################################################################
# Define the function to train the model
# ##############################################################################

#when imported in airflow from.. import * does not work
def __read_input_data(db_path, db_file_name, table_name):
    cnx = sqlite3.connect(db_path+db_file_name)
    df = pd.read_sql('select * from '+table_name,cnx)
    
    print('df columns at initial read of Training Pipeline', df.columns)
    #df.drop(columns=["level_0",'index'],axis=1, inplace=True,errors="ignore")
    cnx.close()
    print("Data from the ",table_name)
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
        **NOTE : You can modify the encode_featues function used in heart disease's inference
        pipeline for this.

    OUTPUT
        1. Save the encoded features in a table - features

    SAMPLE USAGE
        encode_features()
    '''
    input_data = __read_input_data(db_path,db_file_name, "interactions_mapped")
    
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
    
    #write to database table
    __save_data_to_db(db_path, db_file_name, encoded_df, 'features_inf')
    
    print('Feature_inf Table created in DB : Success')
###############################################################################
# Define the function to load the model from mlflow model registry
# ##############################################################################
# logged_model = 'runs:/a46a9cbe388544ce8448db33d6c6c356/models'

def get_models_prediction(db_path, db_file_name, model_file_path, model_name, model_stage, tracking_uri):
    '''
    This function loads the model which is in production from mlflow registry and 
    uses it to do prediction on the input dataset. Please note this function will the load
    the latest version of the model present in the production stage. 

    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be
        model from mlflow model registry
        model name: name of the model to be loaded
        stage: stage from which the model needs to be loaded i.e. production


    OUTPUT
        Store the predicted values along with input data into a table

    SAMPLE USAGE
        load_model()
    '''
    cnx = sqlite3.connect(db_path+db_file_name)
    
    #read target & featuers
    X = pd.read_sql('select * from features_inf',cnx)
    print('Feature_inf Variable', X.columns)
    print('Feature_inf shape', X.shape)
    
    mlflow.set_tracking_uri(tracking_uri)
    model_production_uri = "models:/{model_name}/production".format(model_name=model_name)
    print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_production_uri))
    
    model_production = mlflow.pyfunc.load_model(model_production_uri)
    
    #loaded_model = mlflow.sklearn.load_model(model_file_path)
    # # Predict on a Pandas DataFrame.
    predictions = model_production.predict(pd.DataFrame(X))
    print('predictions shape', predictions.shape)
    print('predictions head', predictions)
    print('predictions loaded_model shape', model_production)
    
    df = pd.DataFrame(predictions,columns=["app_complete_flag_pred"])
    print(df["app_complete_flag_pred"].value_counts())
    
    merge_df = pd.DataFrame(data = X)
    merge_df['app_complete_flag_pred'] = predictions
    print('merge_df : ', merge_df.columns)
    
    __save_data_to_db(db_path, db_file_name, merge_df, 'FINAL_PREDICTION')


###############################################################################
# Define the function to check the distribution of output column
# ##############################################################################

def prediction_ratio_check(db_path, db_file_name):
    '''
    This function calculates the % of 1 and 0 predicted by the model and  
    and writes it to a file named 'prediction_distribution.txt'.This file 
    should be created in the ~/airflow/dags/Lead_scoring_inference_pipeline 
    folder. 
    This helps us to monitor if there is any drift observed in the predictions 
    from our model at an overall level. This would determine our decision on 
    when to retrain our model.
    

    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be

    OUTPUT
        Write the output of the monitoring check in prediction_distribution.txt with 
        timestamp.

    SAMPLE USAGE
        prediction_col_check()
    '''
    cnx = sqlite3.connect(db_path+db_file_name)
    
    #read FINAL_PREDICTION
    df = pd.read_sql('select * from FINAL_PREDICTION',cnx)
    print('FINAL_PREDICTION columns', df.columns)
    
    print(df.app_complete_flag_pred.value_counts(normalize=True).mul(100).round(1).astype(str) + '%')
    
    # Append-adds at last
    file1 = open("dags/Lead_scoring_inference_pipeline/prediction_distribution.txt", "a")  # append mode
    
    # datetime object containing current date and time
    now = datetime.now()
    print("now =", now)

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("date and time =", dt_string)
    
    file1.write("Prediction DateTime :"+dt_string+"\n")
    df = df.app_complete_flag_pred.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
    file1.write(df.to_string())
    file1.write("\n\n\n")
    file1.close()
###############################################################################
# Define the function to check the columns of input features
# ##############################################################################


def input_features_check(db_path, db_file_name, one_hot_encoded_features):
    '''
    This function checks whether all the input columns are present in our new
    data. This ensures the prediction pipeline doesn't break because of change in
    columns in input data.

    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be
        ONE_HOT_ENCODED_FEATURES: List of all the features which need to be present
        in our input data.

    OUTPUT
        It writes the output in a log file based on whether all the columns are present
        or not.
        1. If all the input columns are present then it logs - 'All the models input are present'
        2. Else it logs 'Some of the models inputs are missing'

    SAMPLE USAGE
        input_col_check()
    '''
    cnx = sqlite3.connect(db_path+db_file_name)
    
    #read target & featuers
    X = pd.read_sql('select * from features_inf',cnx)
    
    #get all cloumns of the tables in list
    X_cols = X.columns.to_list()
    
    print('columns from the source table',X_cols)
    print('columns from the raw_data_schema',one_hot_encoded_features)
    
    if collections.Counter(X_cols) == collections.Counter(one_hot_encoded_features):
        print('All the models input are present')
    else:
        print('Some of the models inputs are missing')