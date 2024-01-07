import logging
import pandas as pd
from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from model.data_cleaning import DataCleaning, DataDivideStrategy, DataPreprocessingStrategy

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, 'X_train'],
    Annotated[pd.DataFrame, 'X_test'],
    Annotated[pd.Series, 'y_train'],
    Annotated[pd.Series, 'y_test'],
]:
    '''
    Cleans the data and divides it into train and test sets

    Args: 
        df: raw data
    Returns:
        X_train: Training data
        X_test: Test data
        y_train: Training labels
        y_test: Test labels
    '''
    try:
        process_strategy = DataPreprocessingStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()

        dividing_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, dividing_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info('Data Cleaning Completed')
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f'Error in Data Cleaning: {e}')
        raise e

