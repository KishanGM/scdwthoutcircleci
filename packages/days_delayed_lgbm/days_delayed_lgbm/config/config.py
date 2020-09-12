import pathlib
import pandas as pd


import days_delayed_lgbm




PACKAGE_ROOT = pathlib.Path(days_delayed_lgbm.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "datasets"

# data
TESTING_DATA_FILE = "test.csv"
TRAINING_DATA_FILE = "train.csv"
TARGET = 'late_delivery'
ID_COL = 'order_id'
Y_TRAIN = []

pd.options.display.max_rows = 10
pd.options.display.max_columns = 10


# variables
FEATURES =    [  #'order_id'
                 #,'late_delivery'
                 'customer_country','customer_segment'
                 ,'samecountry_source_dest'
                 ,'ordered_on_weekends'
                 ,'market'
                 ,'shipping_mode'
                 ,'order_dt_month','store_country','order_country','order_region'
                 ,'order_dt_weekday','cat_lowvol_lowrisk','cat_lowvol_highrisk', 'cat_highvol_lowrisk',
                'lowvol_lowrisk', 'highvol_lowrisk',
                'lowvol_highrisk', 'highvol_highrisk', 'order_country_logistics_performance_index'

]

# categorical variables with NA in train set
MEAN_ENC_COLS = [
                 'customer_country','customer_segment'
                 ,'samecountry_source_dest'
                 ,'ordered_on_weekends'
                 ,  'market'
                 , 'shipping_mode'
                 ,'order_dt_month','store_country','order_country','order_region'
                 ,'order_dt_weekday','cat_lowvol_lowrisk','cat_lowvol_highrisk', 'cat_highvol_lowrisk',
                'lowvol_lowrisk', 'highvol_lowrisk',
                'lowvol_highrisk', 'highvol_highrisk', 'order_country_logistics_performance_index'

]

LGB_PARAMS = {
    'max_bin': 100,
    'objective': 'binary',
    'learning_rate': 0.01,
    'n_estimators': 1000,
    'boosting_type': 'gbdt',
    'metric': ['auc', 'binary_logloss',],
    'boost_from_average': True
}

EVAL_RESULTS = {}

PIPELINE_NAME = "lgbm_pipeline"
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}_output_v"

# used for differential testing
ACCEPTABLE_MODEL_DIFFERENCE = 0.05
