import numpy as np
import pandas as pd
import math
import json
import logging
import joblib

from flask_restful import abort

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM as lstm
from keras.layers import GRU
from sklearn.preprocessing import MinMaxScaler

from IForecastingModel import IForecastingModel, create_model_file_name


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def create_and_train(train_X, train_y, test_X, test_y, epochs, batch_size, hidden_l, scaler, model_type='LSTM'):
    lag = train_X.shape[1]
    num_classes = train_X.shape[2]

    # design network
    model = Sequential()
    if model_type == 'LSTM':
        model.add(lstm(hidden_l, input_shape=(lag, num_classes)))
    else:
        model.add(GRU(hidden_l, input_shape=(lag, num_classes)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    try:
        model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_X, test_y), verbose=2,
                  shuffle=False)
    except Exception:
        return {'error': -1,
                'epochs': epochs,
                'batch_size': batch_size,
                'hidden_l': hidden_l,
                'lag': lag}

    if test_X is None:
        return {'error': -1,
                'model': model,
                'epochs': epochs,
                'batch_size': batch_size,
                'hidden_l': hidden_l,
                'lag': lag}
    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], lag * num_classes))
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, -(num_classes-1):]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, -(num_classes-1):]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # calculate RMSE
    error = IForecastingModel.evaluate_forecast(inv_y, inv_yhat)
    return {'error': error,
            'model': model,
            'epochs': epochs,
            'batch_size': batch_size,
            'hidden_l': hidden_l,
            'lag': lag}


def prepare_data(train_ds, test_ds, lags):
    train_ds = train_ds.drop(['Datetime'], axis=1)
    train_values = train_ds.values

    if test_ds is not None:
        test_ds = test_ds.drop(['Datetime'], axis=1)
        test_values = test_ds.values
        ds = np.concatenate((train_values, test_values))
        train_size = len(train_values)
    else:
        ds = train_ds
        train_size = int(len(train_values) * 0.8)

    # ensure all data is float
    values = ds.astype('float32')
    # normalize features
    values = np.reshape(values, (len(values), -1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # specify number of features
    n_features = values.shape[1]
    # frame as supervised learning
    reframed = series_to_supervised(scaled, lags, 1)

    # split into train and test sets
    values = reframed.values

    train = values[:train_size, :]
    test = values[train_size:, :]
    # split into input and outputs
    n_obs = lags * n_features
    train_X, train_y = train[:, :n_obs], train[:, -n_features]
    test_X, test_y = test[:, :n_obs], test[:, -n_features]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], lags, n_features))
    test_X = test_X.reshape((test_X.shape[0], lags, n_features))

    return test_X, test_y, train_X, train_y, scaler


def prepare_parameters(parameters, seasonality):
    n_epochs = [10, 25, 50]
    n_batch_size = [20, 30, 50]
    n_hidden_l = [30, 40, 50]

    if 'epochs' in parameters:
        user_epochs = parameters['epochs']
        if len(user_epochs) >= 1:
            n_epochs = user_epochs
    if 'batch_size' in parameters:
        user_batch_size = parameters['batch_size']
        if len(user_batch_size) >= 1:
            n_batch_size = user_batch_size
    if 'hidden_l' in parameters:
        user_hidden_l = parameters['hidden_l']
        if len(user_hidden_l) >= 1:
            n_hidden_l = user_hidden_l

    # lag is set to seasonal period to cover one complete period
    if len(seasonality) == 0:
        lag = 1
    else:
        lag = min(seasonality)
    if lag < 1:
        lag = 1

    lag = math.ceil(lag)

    return n_epochs, n_batch_size, n_hidden_l, lag


def save_model(best_model, model_type, columns, scaler, last_values):
    path = create_model_file_name(model_type, "h5")
    best_model['model'].save(path)
    info_file = path[:-3] + ".json"
    # save important infos like the lag (needed for prediction
    info = {
        'lag': best_model['lag'],
        'columns': columns,
        'last_values': last_values
    }
    with open(info_file, 'w') as file:
        json.dump(info, file)
    scaler_path = path[:-3] + ".pkl"
    joblib.dump(scaler, scaler_path)

    return path


def train_multi_threaded(train_X, train_y, test_X, test_y, n_epochs, n_batch_size, n_hidden_l, scaler, model_type):
    threads = []
    results = []

    """"
    # Multithreading disabled, crash with no Error message during stress test (multiple datasets directly after each other)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for epochs in n_epochs:
            for batch_size in n_batch_size:
                for hidden_l in n_hidden_l:
                    threads.append(executor.submit(create_and_train, train_X, train_y, test_X, test_y, epochs,
                                                   batch_size, hidden_l, scaler, model_type))
                                                   

        for i in range(len(threads)):
            results.append(threads[i].result())
    """

    for epochs in n_epochs:
        for batch_size in n_batch_size:
            for hidden_l in n_hidden_l:
                result = create_and_train(train_X, train_y, test_X, test_y, epochs, batch_size, hidden_l, scaler, model_type)
                results.append(result)

    return results


def get_bet_model(results):
    current_best = None
    for result in results:
        if result['error'] != -1:
            if current_best is None or current_best['error'] > result['error']:
                current_best = result
    return current_best


# source: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
class LSTM(IForecastingModel):

    @staticmethod
    def is_multivariate():
        return True

    @staticmethod
    def create_and_train_model(train_ds, test_ds, parameters, seasonality):

        n_epochs, n_batch_size, n_hidden_l, lag = prepare_parameters(parameters, seasonality)

        train_X, train_y, test_X, test_y, scaler = prepare_data(train_ds, test_ds, lag)

        logging.getLogger(__name__).info("Start LSTM training")
        results = train_multi_threaded(train_X=train_X, train_y=train_y, test_X=test_X, test_y=test_y,
                                       n_epochs=n_epochs, n_batch_size=n_batch_size, n_hidden_l=n_hidden_l,
                                       scaler=scaler, model_type='LSTM')

        best_model = get_bet_model(results)

        logging.getLogger(__name__).info("Finished LSTM training")

        if best_model is None:
            return None, -1, None

        params = {
            'lag': lag,
            'epochs': best_model['epochs'],
            'batch_size': best_model['batch_size'],
            'hidden_layer': best_model['hidden_l']
        }

        if len(train_ds.columns) > 2:
            last_values = None
        else:
            # save the last values of test_X if the model is univariate
            last_values = list(test_X[-1:].flatten())
            last_values = [float(i) for i in last_values]

        path = save_model(best_model, 'lstm', len(train_ds.columns), scaler, last_values)

        error = {
            'rmse': best_model['error'],
        }

        return path, error, params

    @staticmethod
    def create_forecast(model_path, period, misc):
        info_path = model_path[:-3] + ".json"

        with open(info_path, 'r') as file:
            info = json.load(file)

        model = load_model(model_path)

        scaler_path = model_path[:-3] + ".pkl"
        scaler = joblib.load(scaler_path)

        columns = info['columns'] - 1
        lag = info['lag']

        ds = misc['dataset']

        own_ds = False
        if columns is 1 and ds is None:
            last_values = info['last_values']
            ds_values = last_values + ([0] * period)
            ds = pd.DataFrame(ds_values, columns=['values'])
            own_ds = True

        if ds is None:
            abort(400, message=f"A dataset in the csv format with the to predict amount of lines and {columns} columns is "
                               f"required for this model. The first column can be set to 0 with the exception of the "
                               f"first {lag} values. ")

        # ensure all data is float
        for i in range(1, len(ds.columns)):
            ds[ds.columns[i]] = ds.values[:, i].astype(float)
        values = ds.values.astype('float32')
        # normalize features
        values = np.reshape(values, (len(values), -1))
        prep_ds = scaler.transform(values)

        for i in range(len(prep_ds)-lag):
            pred_step = []
            for j in range(lag):
                pred_step.append(prep_ds[i+j])

            pred = model.predict(np.asarray([pred_step]))
            prep_ds[i+lag][0] = pred[0][0]

        # prep_ds = prep_ds.reshape((prep_ds.shape[0], lag * columns))
        inv_yhat = scaler.inverse_transform(prep_ds)
        inv_yhat = inv_yhat[:, 0]

        prediction = list(inv_yhat.astype(float))

        # remove last training values of prediction with period set
        if own_ds is True:
            prediction = prediction[lag:]

        return prediction
