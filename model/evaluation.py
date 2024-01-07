import logging
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    '''
    Abstract class defining strategy for evaluate our models
    '''
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        '''
        Calculates the scores of the model

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            None
        '''
        pass


class MSE(Evaluation):
    '''
    Evaluation strategy that uses Mean Squared Error
    '''
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try: 
            logging.info('Calculating MSE')
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f'MSE: {mse}')
            return mse
        except Exception as e:
            logging.error(f'Error in calculating MSE: {e}')
            raise e
        

class R2(Evaluation):
    '''
    Evaluation strategy that uses R2
    '''
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try: 
            logging.info('Calculating R2 Score')
            r2 = r2_score(y_true, y_pred)
            logging.info(f'R2 Score: {r2}')
            return r2
        except Exception as e:
            logging.error(f'Error in calculating R2: {e}')
            raise e
        

class RMSE(Evaluation):
    '''
    Evaluation strategy that uses Root Mean Squared Error
    '''
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try: 
            logging.info('Calculating RMSE')
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info(f'RMSE: {rmse}')
            return rmse
        except Exception as e:
            logging.error(f'Error in calculating RMSE: {e}')
            raise e