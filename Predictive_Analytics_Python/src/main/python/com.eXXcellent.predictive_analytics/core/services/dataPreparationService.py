import numpy as np
import pandas as pd
import logging

from flask_restful import abort
from collections import Counter
from statsmodels.tsa.seasonal import seasonal_decompose

from siml.detect_peaks import detect_peaks
from scipy.signal import savgol_filter


"""
S: Second,
min: Minute,
H: Hour,
D: Day,
W: Week,
M: Month,
Q: Quarter,
A: Year
"""
valid_time_intervals = ['S', 'min', 'H', 'D', 'W', 'M', 'Q', 'A']


def convert_columns_from_string(columns):
    if columns is None or not columns:
        int_columns = [0, 1]
    else:
        try:
            int_columns = list(map(int, columns.split(';')))
        except ValueError:
            abort(400, message=f"Please separate the columns with ';'")
    return int_columns


def detect_missing_entries(df):
    variance = 0.1      # +- 10 % variance for the default interval
    df.reset_index(inplace=True, drop=True)

    interval = []
    for x in range(10):
        interval.append(df['Datetime'][x + 1] - df['Datetime'][x])
    interval_sorted = Counter(interval).most_common()

    if interval_sorted[0][1] >= 5:
        default_interval = interval_sorted[0][0]
    else:
        default_interval = np.mean(interval)
        logging.getLogger(__name__).warning('Time interval irregular')

    min_interval = default_interval * (1 - variance)        # - 10% variance
    max_interval = default_interval * (1 + variance)        # + 10% variance

    logging.getLogger(__name__).debug(f'default time interval: {default_interval}')
    logging.getLogger(__name__).debug(f'minimum time interval: {min_interval}')
    logging.getLogger(__name__).debug(f'maximum time interval: {max_interval}')

    missing_entries = []

    # iterate through the dataset from the first to the second last element
    for x in range(len(df) - 2):
        if ((df['Datetime'][x + 1] - df['Datetime'][x]) < min_interval) or (
                (df['Datetime'][x + 1] - df['Datetime'][x]) > max_interval):
            expected_entry = df['Datetime'][x] + default_interval
            missing_entries.append(expected_entry)

            difference = df['Datetime'][x + 1] - expected_entry
            if difference > max_interval:
                next_entry = expected_entry + default_interval
                while next_entry < df['Datetime'][x + 1]:
                    missing_entries.append(next_entry)
                    next_entry = next_entry + default_interval

            # TODO Zeitumstellung

    return missing_entries


def insert_missing_dates_in_dataframe(dataframe, missing_dates):
    for missing_date in missing_dates:
        index = len(dataframe)
        dataframe.loc[index] = np.NaN
        dataframe.loc[index, 'Datetime'] = pd.to_datetime(missing_date, format='%d.%m.%Y %H:%M:%S')

    dataframe.sort_values(inplace=True, by='Datetime')
    dataframe.reset_index(inplace=True, drop=True)
    return dataframe


def resample_dataframe(dataframe, new_interval='D'):
    """
    resamples a dataframe
    :param dataframe: The dataframe to resample
    :param new_interval: The new Time interval the dataframe should have:
        S: Second,
        min: Minute,
        H: Hour,
        D: Day,
        W: Week,
        M: Month,
        Q: Quarter,
        A: Year,
    :return: The resampled dataframe
    """
    df = dataframe.set_index('Datetime', inplace=False).resample(new_interval).sum()
    df.reset_index(inplace=True)
    return df


def interpolate_dataframe(dataframe, method="linear"):
    """
    interpolates missing values in the time series
    :param dataframe: dataframe to interpolate
    :param method: method used for interpolation
        linear,
        ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘spline’, ‘barycentric’, ‘polynomial’
    :return: interpolated dataframe
    """
    return dataframe.interpolate(method=method)


def decompose_dataframe(dataframe, interval, model='additive'):
    df = resample_dataframe(dataframe, interval)
    df = df.set_index('Datetime')
    return seasonal_decompose(df[df.keys()[0]], model=model)


def find_seasonality(df, interval):
    # https://ataspinar.com/2020/12/22/time-series-forecasting-with-stochastic-signal-analysis-techniques/

    logging.info("This is working somehow")

    df_trend = savgol_filter(df, 25, 1)
    df_detrended = df - df_trend

    fft_y_ = np.fft.fft(df_detrended)
    fft_y = np.abs(fft_y_[:len(fft_y_) // 2])
    mph = np.nanmax(fft_y) * 0.4
    indices_peaks = detect_peaks(fft_y, mph=mph)

    """
    fft_x_ = np.fft.fftfreq(len(df))
    fft_x = fft_x_[:len(fft_x_) // 2]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(fft_x, fft_y)
    ax.scatter(fft_x[indices_peaks], fft_y[indices_peaks], color='red', marker='D')
    ax.set_title('frequency spectrum of Air Passengers dataset')
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Frequency')

    for idx in indices_peaks:
        x, y = fft_x[idx], fft_y[idx]
        text = "  f = {:.2f}".format(x, y)
        ax.annotate(text, (x, y))
    plt.savefig("debug/season.png")
    """

    found_intervals = len(df) / indices_peaks
    if interval is None:
        if len(found_intervals) > 10:
            return []
        logging.getLogger(__name__).info(f'Found intervals: {found_intervals}')
        return found_intervals
    time_interval_table = pd.read_table("./misc/time_intervals.csv", delimiter=';')
    intervals_available = time_interval_table[interval].values.astype(np.float)

    reasonable_intervals = []
    for found_interval in found_intervals:
        min = found_interval * 0.9
        max = found_interval * 1.1
        for interval_candidate in intervals_available:
            if min < interval_candidate < max:
                reasonable_intervals.append(interval_candidate)

    # remove duplicates
    reasonable_intervals = list(dict.fromkeys(reasonable_intervals))
    logging.getLogger(__name__).info(f'Found intervals: {reasonable_intervals}')
    return reasonable_intervals


def analyse_prepare_data(df, target_interval, missing=True, decompose=False, seasonality=True):
    if target_interval not in valid_time_intervals:
        abort(422, message=f"The 'interval' parameter is not valid. Value: {target_interval} not valid. Valid values: {valid_time_intervals}")

    time_range = df['Datetime'][len(df) - 1] - df['Datetime'][0]
    logging.getLogger(__name__).debug(f'Time range: {time_range}')

    if missing is True:
        missing_dates = detect_missing_entries(df)
        logging.getLogger(__name__).debug(f'missing entries detected:{len(missing_dates)}: {missing_dates}')
        df = insert_missing_dates_in_dataframe(df, missing_dates)
        df = interpolate_dataframe(df, 'linear')

    logging.getLogger(__name__).debug(f'original entries number:{len(df)}')
    df = resample_dataframe(df, target_interval)
    logging.getLogger(__name__).debug(f'resampled entries number:{len(df)}')

    if decompose is True:
        decomposed = decompose_dataframe(df, interval=target_interval)
        decomposed.plot()
        trend_component = decomposed.trend
        seasonal_component = decomposed.seasonal
        resid_component = decomposed.resid
        # For possible later use, send to user for example
        # plt.savefig("debug/test.png")
    else:
        trend_component = []
        seasonal_component = []
        resid_component = []

    if seasonality is True:
        values = df[df.columns[1]].values
        seasonal_periods = find_seasonality(values, target_interval)
    else:
        seasonal_periods = []

    return {
        'df': df,
        'seasonality': seasonal_periods,
        'trend': trend_component,
        'seasonal': seasonal_component,
        'random error': resid_component
    }
