import logging
import pickle

from pmdarima.arima import auto_arima

from IForecastingModel import IForecastingModel, create_model_file_name


class SARIMA(IForecastingModel):

    @staticmethod
    def create_forecast(model_path, period, misc):
        model = SARIMA.load_model(model_path)
        return list(model.predict(n_periods=period))

    @staticmethod
    def create_and_train_model(train_ds, test_ds, parameters, seasonality):
        logging.getLogger(__name__).info("start arima training")

        train_values = train_ds[train_ds.columns[1]].values

        if 'p' in parameters or 'd' in parameters or 'q' in parameters or 'P' in parameters or 'D' in parameters or 'Q' in parameters:
            p = 0 if 'p' not in parameters else parameters['p']
            d = 0 if 'd' not in parameters else parameters['d']
            q = 0 if 'q' not in parameters else parameters['q']
            P = 0 if 'P' not in parameters else parameters['P']
            D = 0 if 'D' not in parameters else parameters['D']
            Q = 0 if 'Q' not in parameters else parameters['Q']
            model = auto_arima(train_values, start_p=p, d=d, start_q=q, max_p=p, max_d=d, max_q=q,
                               start_P=P, D=D, start_Q=Q, max_P=P, max_D=D, max_Q=Q)
        else:
            if len(seasonality) == 0:
                m = 1
            else:
                m = min(seasonality)
            if m <= 0:
                m = 1
            model = auto_arima(train_values, start_p=0, d=0, start_q=0, start_P=0, D=0, start_Q=0, n_fits=50, m=m, seasonal=True)

        logging.getLogger(__name__).info("finished arima training")
        path = create_model_file_name("arima", "pkl")
        SARIMA.save_model(model, path)

        if test_ds is None:
            error = -1
        else:
            test_values = test_ds[test_ds.columns[1]].values
            n_periods = len(test_values)
            prediction = model.predict(n_periods=n_periods)
            error = IForecastingModel.evaluate_forecast(prediction, test_values)

        params = model.get_params(False)
        errors = {
            'aic': model.aic(),
            'bic': model.bic(),
            'rmse': error,
        }

        return path, errors, params

    @staticmethod
    def save_model(model, path):
        with open(path, 'wb') as pkl:
            pickle.dump(model, pkl)

    @staticmethod
    def load_model(path):
        with open(path, 'rb') as pkl:
            return pickle.load(pkl)
