import logging

from IForecastingModel import IForecastingModel
from LSTM import prepare_data, prepare_parameters, save_model, train_multi_threaded, get_bet_model
from LSTM import LSTM


class GRU(IForecastingModel):

    @staticmethod
    def is_multivariate():
        return True

    @staticmethod
    def create_and_train_model(train_ds, test_ds, parameters, seasonality):
        n_epochs, n_batch_size, n_hidden_l, lag = prepare_parameters(parameters, seasonality)

        train_X, train_y, test_X, test_y, scaler = prepare_data(train_ds, test_ds, lag)

        logging.getLogger(__name__).info("Start GRU training")

        results = train_multi_threaded(train_X=train_X, train_y=train_y, test_X=test_X, test_y=test_y,
                                       n_epochs=n_epochs, n_batch_size=n_batch_size, n_hidden_l=n_hidden_l,
                                       scaler=scaler, model_type='GRU')

        logging.getLogger(__name__).info("Finished GRU training")

        error_value = []
        for result in results:
            error_value.append(result['error'])
        print(error_value)

        best_model = get_bet_model(results)

        if best_model is None:
            return None, -1, None

        path = save_model(best_model, 'gru', len(train_ds.columns), scaler)

        params = {
            'lag': lag,
            'epochs': best_model['epochs'],
            'batch_size': best_model['batch_size'],
            'hidden_layer': best_model['hidden_l']
        }

        error = {
            'rmse': best_model['error'],
        }

        return path, error, params

    @staticmethod
    def create_forecast(model_path, period, misc):
        return LSTM.create_forecast(model_path, period, misc)
