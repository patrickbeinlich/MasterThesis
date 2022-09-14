import logging
import pickle
import concurrent.futures

from statsmodels.tsa.holtwinters import ExponentialSmoothing

from IForecastingModel import IForecastingModel, create_model_file_name


def create_and_train(train_ds, trend, seasonal, seasonality):
    # negative dataset values case a value error if trend or seasonal is set to 'mul'
    try:
        model = ExponentialSmoothing(endog=train_ds, trend=trend, seasonal=seasonal, seasonal_periods=int(seasonality)).fit()
    except ValueError:
        return {
            'model': None,
            'params': f"trend: {trend}, seasonal: {seasonal}, seasonality: {seasonality}"
        }

    return {
        'model': model,
        'params': {
            'trend': trend,
            'seasonal': seasonal,
            'seasonality': seasonality
        }
    }


class ES(IForecastingModel):

    @staticmethod
    def load_es_model(path):
        with open(path, 'rb') as file:
            model = pickle.load(file)

        return model

    @staticmethod
    def create_and_train_model(train_ds, test_ds, parameters, seasonality):
        if len(seasonality) == 0:
            period = 0
        else:
            period = min(seasonality)
        if period < 1:
            period = 0

        logging.getLogger(__name__).info("start ES training")

        options = [None, 'add', 'mul']

        results = []

        threads = []
        train_values = train_ds[train_ds.columns[1]].values
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for trend in options:
                for seasonal in options:
                    threads.append(executor.submit(create_and_train, train_values, trend, seasonal, period))

            for thread in threads:
                results.append(thread.result())

        logging.getLogger(__name__).info("finished ES training")

        current_best = None
        for result in results:
            if result['model'] is not None:
                if current_best is None or current_best['model'].aic > result['model'].aic:
                    current_best = result

        current_best_model = current_best['model']

        path = create_model_file_name("es", 'pkl')
        with open(path, 'wb') as file:
            pickle.dump(current_best_model, file)

        if test_ds is None:
            error = -1
        else:
            test_values = test_ds[test_ds.columns[1]].values
            prediction = current_best_model.forecast(len(test_values))
            error = IForecastingModel.evaluate_forecast(prediction, test_values)

        errors = {
            'aic': current_best_model.aic,
            'bic': current_best_model.bic,
            'rmse': error,
        }

        return path, errors, current_best['params']

    @staticmethod
    def create_forecast(model_path, period, misc):
        model = ES.load_es_model(model_path)
        return list(model.forecast(period))
