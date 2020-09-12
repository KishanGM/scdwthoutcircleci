import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib


from days_delayed_lgbm import pipeline
from days_delayed_lgbm.config import config



def save_pipeline(*, pipeline_to_persist) -> None:
    """Persist the pipeline."""
    save_file_name = "days_delayed_lgbm_model.pkl"
    save_path = config.TRAINED_MODEL_DIR / save_file_name
    joblib.dump(pipeline_to_persist, save_path)
    print("saved pipeline")


def run_training() -> None:
    """Train the model."""
    DFTrainTest = pd.read_csv(config.DATASET_DIR / config.TRAINING_DATA_FILE)
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
    save_pipeline(pipeline_to_persist=pipeline.days_delayed_pipe)

if __name__ == '__main__':
    run_training()
