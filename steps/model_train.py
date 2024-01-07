import logging
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from model.model_dev import LinearRegressionModel
from .config import ModelNameConfig
@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig,
) -> RegressorMixin:
    '''
    Trains the model on the ingested data

    Args:
        X_train (pd.DataFrame): The training input data.
        X_test (pd.DataFrame): The testing input data.
        y_train (pd.Series): The training target data.
        y_test (pd.Series): The testing target data.

    Returns:
        LinearRegressionModel: The trained linear regression model.
    '''
    try:
        model = None
        if config.model_name == 'LinearRegression':
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f'Invalid model name: {config.model_name}')
    except Exception as e:
        logging.error(f'Error in Model Training: {e}')
        raise e