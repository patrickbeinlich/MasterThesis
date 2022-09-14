import os
import pandas as pd
import time
import numpy as np

from flask import current_app, Response
from flask_restful import Resource

from core.services.fileService import read_dataframe_from_csv
from core.services.modelService import create_model_with_df, get_model
from core.services.dataPreparationService import find_seasonality


def read_datasets():
    root_path = "debug/data"
    files = os.listdir(root_path)

    dfs = []
    for file in files:
        df = pd.read_csv(os.path.join(root_path, file), sep=';')

        """
        # For datasets with datetime
        df = read_dataframe_from_csv(os.path.join(root_path, file))
        """
        dfs.append(df)

    return dfs, files


class DebugController(Resource):
    def get(self):
        current_app.logger.info("Analyse model performance")

        dfs, files = read_datasets()

        besucher = ['H', 'D']
        covid = ['D', 'W', 'M']
        ekg = [None]
        power = ['D', 'W', 'M']
        tesla = ['D', 'W', 'M']
        water = ['D', 'W', 'M']

        model = 'AR'

        result_df = []

        for i in range(len(dfs)):
            df = dfs[i]
            for resolution in ekg:
                if resolution is not None:
                    df = df.set_index('Datetime', inplace=False).resample(resolution).sum()
                    df.reset_index(inplace=True)


                print(df.head())
                # try:
                values = df[df.columns[1]].values
                seasonality = find_seasonality(values, resolution)
                # except Exception:
                #    seasonality = [0]

                if not any(seasonality):
                    seasonality = [0]

                start = time.time()

                parameters = {
                    'seasonal_period': seasonality,
                    'resolution': resolution,
                }

                # try:
                model_id, error, params = create_model_with_df(model, df, [0], parameters, resolution)

                end = time.time()
                duration = end - start
                current_app.logger.info(f'file: {files[i]}, duration: {duration}')
                current_app.logger.info(f'result {i}: {error}')

                params = error['params']
                if params is None:
                    params = []

                result_df.append({'file': files[i], 'resolution': str(resolution), 'duration': str(duration),
                                  'params': str(params), 'mse': str(error['mse']), 'rmse': str(error['rmse']),
                                  'aic': str(error['aic']), 'bic': str(error['bic'])})

                save_df = pd.DataFrame(result_df)
                save_df.to_csv(f'debug/results/eval_result_{model}_ekg.csv', sep=';')

                # except Exception:
                #    print("ERROR")
                #    result_df.append({'file': files[i], 'resolution': str(resolution), 'duration': str(0),
                #                      'params': str(0), 'mse': str(0), 'rmse': str(0), 'aic': str(0), 'bic': str(0)})

        result_df = pd.DataFrame(result_df)
        result_df.to_csv(f'debug/results/eval_result_{model}_ekg.csv', sep=';')
