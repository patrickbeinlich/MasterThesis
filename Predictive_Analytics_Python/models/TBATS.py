import pickle
import logging

from tbats import TBATS as tbats

from IForecastingModel import IForecastingModel, create_model_file_name


class TBATS(IForecastingModel):

    @staticmethod
    def create_and_train_model(train_ds, test_ds, parameters, seasonality):
        if len(seasonality) == 0:
            period = [0]
        else:
            period = seasonality

        estimator = tbats(seasonal_periods=period)
        logging.getLogger(__name__).info("start arima training")
        train_values = train_ds[train_ds.columns[1]].values
        model = estimator.fit(train_values)

        logging.getLogger(__name__).info("finished arima training")
        path = create_model_file_name("tbats", "pkl")
        TBATS.save_model(model, path)

        if test_ds is None:
            error = -1
        else:
            test_values = test_ds[test_ds.columns[1]].values
            prediction = model.forecast(steps=len(test_values))
            error = IForecastingModel.evaluate_forecast(prediction, test_values)

        params = {
            'alpha': model.params.alpha,
            'beta': model.params.beta,
            'use_box_cox': model.params.components.use_box_cox,
        }

        errors = {
            'aic': model.aic,
            'rmse': error,
        }

        return path, errors, params

    @staticmethod
    def create_forecast(model_path, period, misc):
        model = TBATS.load_model(model_path)
        return list(model.forecast(steps=period))

    @staticmethod
    def save_model(model, path):
        with open(path, 'wb') as pkl:
            pickle.dump(model, pkl)

    @staticmethod
    def load_model(path):
        with open(path, 'rb') as pkl:
            return pickle.load(pkl)

