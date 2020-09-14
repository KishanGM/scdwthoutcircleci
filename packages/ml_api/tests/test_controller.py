from days_delayed_lgbm.config import config as model_config
from days_delayed_lgbm.processing.data_management import load_dataset
from days_delayed_lgbm import __version__ as _version

import json
import math


def test_health_endpoint_returns_200(flask_test_client):
    # When
    response = flask_test_client.get('/health')

    # Then
    assert response.status_code == 200


def test_prediction_endpoint_returns_prediction(flask_test_client):
    # Given
    # Load the test data from the regression_model package
    # This is important as it makes it harder for the test
    # data versions to get confused by not spreading it
    # across packages.
    test_data_all = load_dataset(file_name=model_config.TESTING_DATA_FILE)
    test_data = test_data_all.drop(columns=[model_config.ID_COL,model_config.TARGET])
    post_json = test_data[0:1].to_json(orient='records')

    # When
    response = flask_test_client.post('/v1/predict/days_delayed_lgbm',
                                      json=post_json)

    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)
    prediction = response_json['predictions']
    response_version = response_json['version']
    assert prediction == 1
    assert response_version == _version
