import pandas as pd

from days_delayed_lgbm.config import config

def validate_inputs(input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for unprocessable values."""

    validated_data = input_data.copy()

    # check for NA not seen during training
    if input_data[config.FEATURES].isnull().any().any():
        validated_data = validated_data.dropna(axis=0)

    return validated_data
