import logging
from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin
from abc import ABC, abstractmethod

class Model(ABC):
    '''
    Abstract class for model
    '''
    @abstractmethod
    def train(self, X_train, y_train):
        '''
        Trains the model

        Args:
            X_train: Training data
            y_train: Training labels

        Returns:
            None
        '''
        pass


class LinearRegressionModel(Model):
    '''
    Linear Regression Model
    '''
    def train(self, X_train, y_train, **kwargs):
        '''
        Trains the model
        
        Args:
            X_train: Training data
            y_train: Training labels

        Returns:
            None
        '''
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info('Model Training Completed')
            return reg
        except Exception as e:
            logging.error(f'Error in Model Training: {e}')
            raise e