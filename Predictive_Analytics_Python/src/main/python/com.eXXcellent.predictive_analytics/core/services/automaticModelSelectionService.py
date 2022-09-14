import concurrent.futures
import datetime
import numpy as np
import pandas as pd
import logging

from core.services.fileService import read_dataframe_from_csv
from core.services.infoService import get_available_models, get_available_multivariant_models
from core.services.dataService import analyse_prepare_data
from core.services.modelService import create_model_with_df, create_forecast

from model.dataFile import DataFile


def auto_model_selection(file_id, interval, columns, models, prediction_interval, pred_dataset):

    if models is None and len(columns) > 2:
        available_models = get_available_multivariant_models()
    elif models is None:
        available_models = get_available_models()
    else:
        available_models = models.split(';')
        available_models = [model.strip() for model in available_models]

    file = DataFile.query.get(file_id)
    df = read_dataframe_from_csv(file.filePath, separator=file.separator, columns=columns)
    last_date = df.iloc[-1][0]
    prepared_results = analyse_prepare_data(df, interval)

    df = prepared_results['df']

    parameters = {
        'resolution': interval
    }

    threads = []
    results = []
    rmse = []
    params = []

    logging.getLogger(__name__).info("start model selection")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i in range(len(available_models)):
            threads.append(executor.submit(create_model_with_df, available_models[i], df, parameters, interval))

        for i in range(len(threads)):
            model_id, errors, parameter = threads[i].result()
            results.append(errors)
            rmse.append(errors['rmse'])
            params.append(parameter)

    best_index = np.argmin(np.array(rmse))
    best_model = available_models[best_index]

    logging.getLogger(__name__).info("finished model selection")

    model_id, final_errors, final_params = create_model_with_df(best_model, df, parameters, interval, tt_split=1)

    if prediction_interval is None and pred_dataset is None:
        prediction = []
    else:
        try:
            prediction_steps = int(prediction_interval)
        except ValueError:
            try:
                predict_to = datetime.datetime.strptime(prediction_interval, '%d.%m.%Y %H:%M:%S')
                delta = predict_to - last_date
                prediction_steps = int(delta / pd.Timedelta(1, unit=interval))
            except Exception:
                logging.error(f"Could not convert {prediction_interval} into an integer or Datetime in the format 'dd.mm.YY HH:MM:SS'.")
                prediction_steps = 0

        if prediction_steps is 0:
            prediction = []
        else:
            prediction = create_forecast(model_id, prediction_steps, pred_dataset)

    model_results = []

    for i in range(len(available_models)):
        model_results.append({
            'model_name': available_models[i],
            'errors': results[i],
            'parameters': params[i]
        })

    return {
        'prediction': prediction,
        'final_model': {
            'model_type': best_model,
            'model_id': model_id,
            'errors': final_errors,
            'parameters': final_params
        },
        'training_results': model_results
    }
