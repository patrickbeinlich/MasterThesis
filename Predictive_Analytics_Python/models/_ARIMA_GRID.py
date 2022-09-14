import math
import warnings
import logging

from statsmodels.tsa.arima.model import ARIMA

from IForecastingModel import IForecastingModel


class ARIMA_GRID(IForecastingModel):
    @staticmethod
    def create_and_train_model(train_ds, test_ds, parameters, seasonality):
        ps = range(6)
        ds = range(3)
        qs = range(6)

        Ps = range(3)
        Ds = range(2)
        Qs = range(3)

        current_best = {'error': -1}

        if len(seasonality) == 0:
            M = 0
        else:
            M = min(seasonality)

        for p in ps:
            print(f"----- p = {p}")
            for d in ds:
                for q in qs:
                    print(f"q = {q}")

                    for P in Ps:
                        for D in Ds:
                            for Q in Qs:
                                try:
                                    model = ARIMA(train_ds, order=[p, d, q], seasonal_order=[P, D, Q, M])
                                    logging.getLogger(__name__).debug(
                                        "training arima grid search with p={} d={} q={} P={} D={} Q={}".format(p, d, q,
                                                                                                               P,
                                                                                                               D, Q))

                                    with warnings.catch_warnings():
                                        warnings.filterwarnings("ignore")
                                        model_fit = model.fit()

                                    forecast = model_fit.forecast(len(test_ds))
                                    mse = IForecastingModel.evaluate_forecast(forecast, test_ds)
                                    error = model_fit.aic

                                    logging.getLogger(__name__).debug(
                                        "error {} with p={} d={} q={} P={} D={} Q={}".format(error, p, d, q, P, D, Q))
                                    if current_best['error'] > error or current_best['error'] == -1:
                                        current_best = {'error': error,
                                                        'mse': mse,
                                                        'rmse': math.sqrt(mse),
                                                        'aic': model_fit.aic,
                                                        'bic': model_fit.bic,
                                                        'p': p,
                                                        'd': d,
                                                        'q': q,
                                                        'P': P,
                                                        'D': D,
                                                        'Q': Q,
                                                        'M': M}
                                except Exception as ex:
                                    print(type(ex))
                                    logging.getLogger(__name__).info(
                                        f'error with parameters: ({p},{d},{q})({P},{D},{Q},{M})')

        logging.getLogger(__name__).info('best arima grid search result: {} with p={} d={} q={} P={} D={} Q={}'
                                         .format(current_best['error'], current_best['p'], current_best['d'],
                                                 current_best['q'], current_best['P'],
                                                 current_best['D'], current_best['Q']))

        params = f"({current_best['p']},{current_best['d']},{current_best['q']})({current_best['P']},{current_best['D']},{current_best['Q']},{M})"

        error = {
            'aic': current_best['aic'],
            'bic': current_best['bic'],
            'mse': current_best['mse'],
            'rmse': current_best['rmse'],
            'params': params
        }
        return -1, error

    @staticmethod
    def create_forecast(model_path, period):
        pass
