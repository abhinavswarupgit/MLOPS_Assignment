DB_PATH = '/home/dags/Lead_scoring_data_pipeline/'
DB_FILE_NAME = 'lead_scoring_data_cleaning.db' 

DB_FILE_MLFLOW = 'Lead_scoring_mlflow_production.db'

TRACKING_URI = "http://0.0.0.0:6006"
EXPERIMENT = "Lead_scoring_mlflow_production"

FILE_PATH = 'runs:/098ccb43f20e4607b7ebf76007ffba35/models'

# experiment, model name and stage to load the model from mlflow model registry
MODEL_NAME = 'LightGBM'
STAGE = 'production'
EXPERIMENT = 'Lead_scoring_mlflow_production'

# list of the features that needs to be there in the final dataframe for model training
ONE_HOT_ENCODED_FEATURES = ['total_leads_droppped','referred_lead', 'city_tier', 'first_platform_c_Level8', 'first_platform_c_Level2', 'first_platform_c_others', 'first_platform_c_Level0',
                           'first_platform_c_Level7', 'first_utm_medium_c_others', 'first_utm_medium_c_Level13']


#first_utm_source_c_Level6 

# list of features that need to be one-hot encoded , as these contains data as Level1, Level2, Level3,others
FEATURES_TO_ENCODE = ['first_platform_c', 'first_utm_medium_c', 'first_utm_source_c']