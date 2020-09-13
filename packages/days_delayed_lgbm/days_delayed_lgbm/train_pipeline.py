import numpy as np
import pandas as pd
import joblib


from days_delayed_lgbm import pipeline
from days_delayed_lgbm.config import config
from days_delayed_lgbm import __version__ as _version
from days_delayed_lgbm.processing.data_management import load_dataset, save_pipeline


import logging

_logger = logging.getLogger('days_delayed_lgbm')



def run_training() -> None:
    """Train the model."""
    DFTrainTest = load_dataset(file_name=config.TRAINING_DATA_FILE)#pd.read_csv(config.DATASET_DIR / config.TRAINING_DATA_FILE)
    #DFTrainTest = data.drop(columns = [config.ID_COL])
    DFTrain = DFTrainTest.sample(frac =.8)
    DFTest  = DFTrainTest[(~DFTrainTest.order_id.isin(DFTrain.order_id))]
    X_train = DFTrain.drop(columns = [config.ID_COL,config.TARGET])
    y_train = DFTrain[config.TARGET].values
    X_test = DFTest.drop(columns = [config.ID_COL,config.TARGET])
    y_test = DFTest[config.TARGET].values
    config.Y_TRAIN = y_train
    print(X_train.shape,",",y_train.shape,",",config.Y_TRAIN.shape )
    pipeline.days_delayed_pipe.fit(X_train, y_train)

    _logger.info(f"saving model version : {_version}")
    save_pipeline(pipeline_to_persist=pipeline.days_delayed_pipe)

if __name__ == '__main__':
    run_training()
