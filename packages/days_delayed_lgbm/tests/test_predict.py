import math

from days_delayed_lgbm.config import config
from days_delayed_lgbm.predict import make_prediction
from days_delayed_lgbm.processing.data_management import load_dataset
from days_delayed_lgbm.processing.validation import validate_inputs


def test_make_single_prediction():
    # Given
    test_data_all = load_dataset(file_name='test.csv')
    test_data = test_data_all.drop(columns=[config.ID_COL,config.TARGET])
    single_test_json = test_data[0:1].to_json(orient='records')

    # When
    subject = make_prediction(input_data=single_test_json)

    # Then
    assert subject is not None
    #assert isinstance(subject.get('predictions')[0], int)
    assert subject.get('predictions')[0] == 1

def test_make_multiple_predictions():
    # Given
    test_data_all = load_dataset(file_name='test.csv')
    test_data = test_data_all.drop(columns=[config.ID_COL,config.TARGET])
    original_data_length = len(test_data)
    validated_data_length = len(validate_inputs(test_data))
    multiple_test_json = test_data.to_json(orient='records')

    # When
    subject = make_prediction(input_data=multiple_test_json)

    # Then
    assert subject is not None
    assert len(subject.get('predictions')) == validated_data_length


    #assert len(subject.get('predictions')) != original_data_length
