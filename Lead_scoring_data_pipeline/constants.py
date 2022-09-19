# You can create more variables according to your project. The following are the basic variables that have been provided to you
DB_PATH = '/home/dags/Lead_scoring_data_pipeline/'
DB_FILE_NAME = 'lead_scoring_data_cleaning.db' 
DATA_DIRECTORY = '/home/dags/Lead_scoring_data_pipeline/data/'
INTERACTION_MAPPING = '/home/dags/Lead_scoring_data_pipeline/mapping/'
INDEX_COLUMNS = ['created_date', 'first_platform_c', 'first_utm_medium_c', 'first_utm_source_c', 'total_leads_droppped', 'city_tier', 'referred_lead']
#'app_complete_flag'

LEAD_SCORING_FILE='leadscoring_inference.csv'
TABLE_NAME='loaded_data'