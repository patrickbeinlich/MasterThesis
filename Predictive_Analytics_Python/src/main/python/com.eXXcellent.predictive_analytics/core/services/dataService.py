from flask import send_file, current_app
import ast

from core.services.fileService import read_dataframe_from_csv, safe_dataframe_to_csv, save_file, FileTypes
from core.services.dataPreparationService import *
from core.errors.ErrorMessageGenerator import generate_error_message, ErrorTypes
from model.dataFile import DataFile


def str_to_bool(text):
    # https://datascienceparichay.com/article/python-onvert-string-to-boolean/
    if text == "True":
        return True
    elif text == "False":
        return False
    else:
        return None


def execute_data_action(action, file, args):
    df = read_dataframe_from_csv(file.filePath, separator=file.separator)

    if action == 'analyse':
        missing = str_to_bool(args['find_missing'])
        seasonality = str_to_bool(args['find_seasonality'])
        decompose = str_to_bool(args['decompose'])
        interval = args['interval']

        if interval is None:
            abort(400, message=generate_error_message(ErrorTypes.MissingParameter, 'interval',
                                                      "S, min, H, D, W, M, Q or A"))
        if missing is None:
            missing = True
        if seasonality is None:
            seasonality = True
        if decompose is None:
            decompose = False

        results = analyse_prepare_data(df, interval, missing=missing, decompose=decompose,
                                       seasonality=seasonality)
        safe_dataframe_to_csv(file.filePath, results['df'])
        return {
            'file_id': file.id,
            'seasonality': list(results['seasonality']),
            'trend_component': list(results['trend']),
            'seasonal_component': list(results['seasonal']),
            'random error': list(results['random error'])
        }
    elif action == 'missing_entries':
        missing_entries = detect_missing_entries(df)
        logging.getLogger(__name__).info(f"Found missing entries: {missing_entries}")
        return {
            'file_id': file.id,
            'missing_entries': [timestamp.strftime('%d.%m.%Y %H:%M:%S') for timestamp in missing_entries]
        }
    elif action == 'insert_entries':
        entries = ast.literal_eval(args['missing_dates'])
        entries = [n.strip() for n in entries]
        if len(entries) is 0:
            abort(400, message=generate_error_message(ErrorTypes.MissingParameter, 'missing_dates',
                                                      "['dd.mm.yy HH:MM:SS', ...]"))
        df = insert_missing_dates_in_dataframe(df, entries)
        safe_dataframe_to_csv(file.filePath, df)
        return send_file(file.filePath, as_attachment=True)
    elif action == 'interpolate':
        method = args['interpolation_method']
        if method is None:
            method = 'linear'
        df = interpolate_dataframe(df, method=method)
        safe_dataframe_to_csv(file.filePath, df)
        return send_file(file.filePath, as_attachment=True)
    elif action == 'seasonality':
        interval = args['interval']
        if interval is not None:
            df = resample_dataframe(df, interval)
        values = df[df.columns[1]].values
        seasonality = find_seasonality(values, args['interval'])
        return {
            'file_id': file.id,
            'seasonality': list(seasonality)
        }
    elif action == 'resample':
        interval = args['interval']
        if interval is None:
            abort(400, message=generate_error_message(ErrorTypes.MissingParameter, 'interval',
                                                      "S, min, H, D, W, M, Q or A"))
        df = resample_dataframe(df, interval)
        safe_dataframe_to_csv(file.filePath, df)
        return send_file(file.filePath, as_attachment=True)
    elif action == 'decompose':
        interval = args['interval']
        if interval is None:
            abort(400, message=generate_error_message(ErrorTypes.MissingParameter, 'interval',
                                                      "S, min, H, D, W, M, Q or A"))
        result = decompose_dataframe(df, interval)

        return {
            'file_id': file.id,
            'trend': list(result.trend),
            'seasonal': list(result.seasonal),
            'random error': list(result.resid)
        }
    else:
        abort(405, message=generate_error_message(ErrorTypes.MethodNotFound, action))


def execute_data_action_upload(action, data_file, separator, args):
    if data_file is None:
        abort(400, message=generate_error_message(ErrorTypes.NoFileFound, current_app.config['DATA_FILES_EXTENSIONS']))

    file = save_file(data_file, FileTypes.DATA, {'separator': separator})

    return execute_data_action(action, file, args)


def execute_data_action_id(action, file_id, args):
    file = DataFile.query.get(file_id)

    if file is None:
        abort(404, message=generate_error_message(ErrorTypes.ResourceNotFound, 'Data file', file_id))

    return execute_data_action(action, file, args)
