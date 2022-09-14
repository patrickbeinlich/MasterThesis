import os
import uuid
import pandas as pd
import traceback
import logging

from flask import current_app
from flask_restful import abort
from werkzeug.utils import secure_filename
from datetime import datetime

from core.errors.ErrorMessageGenerator import generate_error_message, ErrorTypes
from model.enums import FileTypes
from model.dataFile import DataFile
from model.predictionModel import PredictionModel
from db.database import db


# https://flask.palletsprojects.com/en/2.0.x/patterns/fileuploads/

def __get_valid_extensions(file_type):
    """
    Get the valid file extensions for a given file type (model, data)
    :param file_type: The given file type (model, data file)
    :return: The allowed extensions for the file type
    """
    if file_type == FileTypes.MODEL:
        return current_app.config['MODEL_FILES_EXTENSIONS']
    elif file_type == FileTypes.DATA:
        return current_app.config['DATA_FILES_EXTENSIONS']
    else:
        abort(501, message=generate_error_message(ErrorTypes.CaseNotImplemented, file_type,
                                                  traceback.format_stack()[-1]))


def __valid_extension(filename, file_type):
    """
    Checks if the filename has a valid extension for the requested file type
    :param filename: The name of the file
    :param file_type: The file type (model, data file)
    :return: True if the file has a valid extension
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in __get_valid_extensions(file_type)


def get_file_key(file_type):
    """
    Get the form-data body key for the requested file type
    :param file_type: The file type (model, data)
    :return: The key to be used in the form data request body.
    """
    if file_type == FileTypes.MODEL:
        return 'model'
    elif file_type == FileTypes.DATA:
        return 'data'
    else:
        abort(501, message=generate_error_message(ErrorTypes.CaseNotImplemented, file_type,
                                                  traceback.format_stack()[-1]))


def __file_save_target_folder(file_type):
    """
    Get the target save folder for the requested file type
    :param file_type: The file type (model, data file)
    :return: The correct directory to save the file to
    """
    if file_type == FileTypes.MODEL:
        return current_app.config['MODEL_FOLDER']
    elif file_type == FileTypes.DATA:
        return current_app.config['UPLOAD_FOLDER']
    else:
        abort(501, message=generate_error_message(ErrorTypes.CaseNotImplemented, file_type,
                                                  traceback.format_stack()[-1]))


def __save_file(file, file_type):
    """
    saves file to the correct folder
    :param file: uploaded file
    :param file_type: type of the file (trained model, data file, ...)
    :return: The path the file was saved to
    """
    if file.filename == '':
        logging.getLogger(__name__).error('No selected file')
        return None
    if file and __valid_extension(file.filename, file_type):
        file_path = os.path.abspath(os.path.join(__file_save_target_folder(file_type), secure_filename(file.filename)))
        file.save(file_path)
        logging.getLogger(__name__).info(f'file uploaded to path {file_path}')
        return file_path
    logging.getLogger(__name__).error(f"No valid file extension for {file_type}: {file.filename}")
    return None


def save_file(file, file_type, parameter):
    """
    Save a file and create a database entry
    :param file: the file to save
    :param file_type: the type of file (model, data file)
    :param parameter: dictionary of parameter for the database entry
    :return: the created model entry
    """
    saved_file_path = __save_file(file, file_type)

    if saved_file_path is None:
        abort(400, message=generate_error_message(ErrorTypes.NoFileFound, get_file_key(file_type),
                                                  __get_valid_extensions(file_type)))
    saved_file = None
    if file_type == FileTypes.MODEL:
        saved_file = PredictionModel(
            modelPath=saved_file_path,
            modelType=parameter['model_type'],
            resolution=parameter['resolution'],
            rmse=parameter['rmse']
        )
    elif file_type == FileTypes.DATA:
        separator = parameter['separator']
        if not separator:
            separator = ';'
        saved_file = DataFile(filePath=saved_file_path, separator=separator)
    else:
        abort(501, message=generate_error_message(ErrorTypes.CaseNotImplemented, file_type,
                                                  traceback.format_stack()[-1]))

    db.session.add(saved_file)
    db.session.commit()

    return saved_file


def get_data_file(file_id):
    return DataFile.query.get(file_id)


def read_dataframe_from_csv(file_path, separator=';', columns=None, header=True):
    """
    Reads the data from a csv file
    :param file_path: The file to read
    :param separator: The separator used in the file
    :param columns: The columns to read from the file
    :param header: Does the file have a header or not
    :return: The data from the file in form of a pandas dataframe
    """

    if columns is None:
        columns = [0, 1]
    else:
        if 0 not in columns:
            columns.insert(0, 0)
    header = 0 if header else None

    dateparse = lambda x: datetime.strptime(x, current_app.config['DATE_FORMAT'])

    df = pd.read_csv(file_path, header=header, sep=separator, parse_dates=[0],
                     date_parser=dateparse)
    df = df.rename(columns={df.columns[0]: 'Datetime'})

    # use only requested columns
    df = df.iloc[:, columns]

    return df


def safe_dataframe_to_csv(file_path, df, separator=';'):
    df.to_csv(file_path, sep=separator, index=False, date_format=current_app.config['DATE_FORMAT'])


def get_prediction_dataset(pred_file):
    if pred_file is not None:
        file_name = f"../../../..//temp/temp_{str(uuid.uuid4())[:8]}.csv"
        pred_file.save(file_name)
        try:
            dataset = pd.read_csv(file_name, sep=";")
        except Exception:
            dataset = None
        os.remove(file_name)
    else:
        dataset = None
    return dataset
