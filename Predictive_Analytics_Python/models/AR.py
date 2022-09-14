import concurrent.futures
import logging
import pickle
import json

from statsmodels.tsa.ar_model import AutoReg

from IForecastingModel import IForecastingModel, create_model_file_name


def create_and_train(train_ds, lags, seasonality):
    seasonal = True if seasonality != 0 else False

    try:
        model = AutoReg(train_ds, lags=lags, seasonal=seasonal, period=int(seasonality))
        model_fit = model.fit()

        aic = model_fit.aic
        return {
            'aic': aic,
            'lags': lags,
            'model': model_fit
        }
    except Exception:
        return {
            'aic': -1,
            'lags': lags
        }


class AR(IForecastingModel):

    @staticmethod
    def save_model(model, path, end):
        with open(path, 'wb') as pkl:
            pickle.dump(model, pkl)

        info_file = path[:-4] + ".json"
        info = {
            'end': end
        }

        with open(info_file, 'w') as file:
            json.dump(info, file)

    @staticmethod
    def load_model(path):
        with open(path, 'rb') as pkl:
            model = pickle.load(pkl)

        info_file = path[:-4] + ".json"
        with open(info_file, 'rb') as file:
            info = json.load(file)
        return model, info

    @staticmethod
    def create_and_train_model(train_ds, test_ds, parameters, seasonality):
        logging.getLogger(__name__).info("start ar training")

        if len(seasonality) == 0:
            period = 0
        else:
            period = min(seasonality)
        if period < 1:
            period = 0

        max_lag = int(period) if period > 15 else 15

        threads = []
        results = []

        current_min = None

        train_values = train_ds[train_ds.columns[1]].values

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for lag in range(1, max_lag+1):
                lags = list(range(1, lag+1))
                threads.append(executor.submit(create_and_train, train_values, lags, period))

            for i in range(len(threads)):
                results.append(threads[i].result())

        for result in results:
            if result['aic'] != -1:
                if current_min is None or current_min['aic'] > result['aic']:
                    current_min = result

        logging.getLogger(__name__).info("finished ar training")
        best_model = current_min['model']
        path = create_model_file_name("ar", "pkl")
        AR.save_model(best_model, path, len(train_values))

        params = {
            'lags': current_min['lags']
        }

        if test_ds is None:
            error = -1
        else:
            test_values = test_ds[test_ds.columns[1]].values
            prediction = best_model.predict(start=len(train_values), end=(len(train_values) + len(test_values) - 1))
            error = IForecastingModel.evaluate_forecast(prediction, test_values)

        errors = {
            'aic': best_model.aic,
            'bic': best_model.bic,
            'rmse': error,
        }

        return path, errors, params

    @staticmethod
    def create_forecast(model_path, period, misc):
        model, info = AR.load_model(model_path)
        start = info['end']
        return list(model.predict(start=start, end=(start + period -1)))
