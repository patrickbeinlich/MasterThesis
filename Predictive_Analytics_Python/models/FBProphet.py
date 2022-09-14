import json
import logging

from flask_restful import abort
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json

from IForecastingModel import IForecastingModel, create_model_file_name


class FBProphet(IForecastingModel):

    @staticmethod
    def is_multivariate():
        return True

    @staticmethod
    def create_forecast(model_path, period, misc):
        model = FBProphet.load_model(model_path)

        ds = misc['dataset']
        resolution = misc['resolution']
        multi = misc['multi']

        if multi is True and ds is None:
            abort(400, message="A dataset is required for predictions on multivariate models")

        if multi is False and ds is None:
            future = model.make_future_dataframe(periods=period)
        else:
            future = ds
            future.rename(columns={future.columns[0]: "ds"}, inplace=True)

        try:
            forecast = model.predict(future)
        except ValueError as error:
            abort(400, message=error.args)

        prediction = forecast[['ds', 'yhat']]

        prediction.rename(columns={'yhat': "prediction"})
        prediction['ds'] = prediction['ds'].dt.strftime('%d.%m.%Y %H:%M:%S')
        # prediction = prediction.values.tolist()
        # flattened = [element for sublist in prediction for element in sublist]
        return prediction.to_dict()

    @staticmethod
    def load_model(path):
        with open(path, 'r') as fin:
            return model_from_json(json.load(fin))

    @staticmethod
    def create_and_train_model(train_ds, test_ds, parameters, seasonality):
        model = Prophet()
        logging.getLogger(__name__).info("Start prophet training")

        train_ds.rename(columns={train_ds.columns[0]: "ds"}, inplace=True)
        train_ds.rename(columns={train_ds.columns[1]: "y"}, inplace=True)

        if test_ds is not None:
            test_ds.rename(columns={test_ds.columns[0]: "ds"}, inplace=True)
            test_ds.rename(columns={test_ds.columns[1]: "y"}, inplace=True)

        additional_columns = list(train_ds.columns)
        additional_columns.remove('ds')
        additional_columns.remove('y')

        for column in additional_columns:
            model.add_regressor(column, standardize=False)

        model.fit(train_ds)
        logging.getLogger(__name__).info("finished prophet training")
        path = create_model_file_name("prophet", "json")
        with open(path, 'w') as fout:
            json.dump(model_to_json(model), fout)

        if test_ds is None:
            rmse = -1
        else:
            future = test_ds.drop('y', axis=1)
            forecast = model.predict(future)
            predicted = forecast['yhat'].values
            rmse = IForecastingModel.evaluate_forecast(predicted, test_ds['y'].values)

        params = {}

        errors = {
            'rmse': rmse
        }

        return path, errors, params
