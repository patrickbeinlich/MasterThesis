import os
import logging

from flask import current_app, send_file
from flask_restful import abort

from db.database import db
from model.predictionModel import PredictionModel
from model.dataFile import DataFile
from core.errors.ErrorMessageGenerator import generate_error_message, ErrorTypes
from core.services.fileService import read_dataframe_from_csv, FileTypes, save_file
from core.services.dataPreparationService import resample_dataframe, find_seasonality
from core.services.infoService import get_module


def create_model_with_file(module, file, columns, parameters, separator, interval):
    if file is None:
        abort(400, message=generate_error_message(ErrorTypes.NoFileFound, current_app.config['DATA_FILES_EXTENSIONS']))

    file = save_file(file, FileTypes.DATA, {'separator': separator})

    return create_model(module, file.id, columns, parameters, interval)


def create_model(module, file_id, columns, parameters, interval):
    file = DataFile.query.get(file_id)
    if file is None:
        abort(404, message=generate_error_message(ErrorTypes.ResourceNotFound, 'Data file', file_id))

    df = read_dataframe_from_csv(file.filePath, separator=file.separator, columns=columns)

    return create_model_with_df(module, df, parameters, interval)


def create_model_with_df(module, df, parameters, interval, tt_split=0.8):
    fc_module = get_module(module)

    if interval is not None:
        df = resample_dataframe(df, interval)

    values = df[df.columns[1]].values
    seasonality = find_seasonality(values, interval)

    if len(df.columns) > 2:
        multivariate = True
    else:
        multivariate = False

    for i in range(1, len(df.columns)):
        df[df.columns[i]] = df.values[:, i].astype(float)

    if tt_split < 1:
        train_size = int(len(df) * tt_split)
        train_ds = df.iloc[:train_size, :]
        test_ds = df.iloc[train_size:, :]
    else:
        train_ds = df
        test_ds = None

    model_path, errors, params = getattr(fc_module, 'create_and_train_model')(train_ds, test_ds, parameters, seasonality)
    model = PredictionModel(modelType=module,  modelPath=model_path, multivariate=multivariate, rmse=errors['rmse'],
                            resolution=interval)
    try:
        db.session.add(model)
        db.session.commit()
        return model.id, errors, params
    except:
        logging.getLogger(__name__).error("Failed to persist model information to database")
        return -1, errors, params


def delete_model(model_id):
    model = PredictionModel.query.get(model_id)

    if os.path.exists(model.modelPath):
        os.remove(model.modelPath)
    else:
        logging.getLogger(__name__).warning("Model file not found")

    db.session.delete(model)
    db.session.commit()

    pass


def create_forecast(model_id, interval, dataset):
    model = PredictionModel.query.get(model_id)
    model_type = model.modelType
    model_path = model.modelPath
    resolution = model.resolution
    multi = model.multivariate

    fc_module = get_module(model_type)

    try:
        interval = int(interval)
    except ValueError:
        abort(400, message="The parameter 'prediction_interval' has to be an integer")

    if multi is True and dataset is None:
        abort(400, message="A dataset is required for multivariate predictions. Please supply a dataset with the "
                           "correct columns under the 'data' key")

    misc = {
        # for LSTM, GRU, and Prophet multivariate
        'dataset': dataset,
        # for Prophet univariate
        'resolution': resolution,
        'multi': multi
    }

    prediction = getattr(fc_module, 'create_forecast')(model_path, interval, misc)

    return {
        'prediction': prediction
    }


def get_model(model_id):
    model = PredictionModel.query.get(model_id)
    return send_file(model.modelPath, as_attachment=True)

