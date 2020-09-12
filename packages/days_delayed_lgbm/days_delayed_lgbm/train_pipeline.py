import pathlib


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent
#TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'datasets'

TESTING_DATA_FILE = DATASET_DIR / 'test.csv'
TRAINING_DATA_FILE = DATASET_DIR / 'train.csv'
TARGET = 'late_delivery'


FEATURES =    [  'order_id'
                 ,'late_delivery'
                 ,'customer_country','customer_segment'
                 ,'samecountry_source_dest'
                 ,'ordered_on_weekends'
                 ,'market'
                 ,'shipping_mode'
                 ,'order_dt_month','store_country','order_country','order_region'
                 ,'order_dt_weekday','cat_lowvol_lowrisk','cat_lowvol_highrisk', 'cat_highvol_lowrisk',
                'lowvol_lowrisk', 'highvol_lowrisk',
                'lowvol_highrisk', 'highvol_highrisk', 'order_country_logistics_performance_index'

]


def save_pipeline() -> None:
    """Persist the pipeline."""

    pass


def run_training() -> None:
    """Train the model."""

    print('Training...')


if __name__ == '__main__':
    run_training()
