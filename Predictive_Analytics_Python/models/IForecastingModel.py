import uuid
import math

from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error


def create_model_file_name(model_type, file_ending):
    return f"../../../../trained_models/{model_type}_{str(uuid.uuid4())[:8]}.{file_ending}"


class IForecastingModel(ABC):

    @staticmethod
    @abstractmethod
    def create_and_train_model(train_ds, test_ds, parameters, seasonality):
        """
        creates and trains a new time series forecasting model
        :param train_ds: training dataset
        :param test_ds: test dataset
        :param parameters: parameter specific for the model
        :param seasonality: seasonality of the dataset
        :return: returns the path where the model file is stored
        """
        return ""

    @staticmethod
    @abstractmethod
    def create_forecast(model_path, period, misc):
        pass

    @staticmethod
    def evaluate_forecast(forecast, original_data):
        return math.sqrt(mean_squared_error(original_data, forecast))

    @staticmethod
    def is_multivariate():
        # Override this function if the model is capable of multivariate forecasting
        """
        Returns, weather the model is capable of multivariate forecasting or not
        :return: If the model is multivariate
        """
        return False


