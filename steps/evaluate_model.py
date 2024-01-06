import logging
import pandas as pd
from zenml import step

@step
def eval_model(df: pd.DataFrame) -> None:
    '''
    Evaluates the model on the ingested data

    Args:
        df: ingested data
    '''
    logging.info('Evaluating the model')
    pass