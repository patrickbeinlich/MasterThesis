import werkzeug.datastructures
import json

from flask import current_app
from flask_restful import Resource, reqparse, abort

from core.services import modelService
from core.services.fileService import save_file, FileTypes, get_file_key, get_prediction_dataset
from core.services.dataPreparationService import convert_columns_from_string

model_args = reqparse.RequestParser()
model_args.add_argument(get_file_key(FileTypes.MODEL), required=False, type=werkzeug.datastructures.FileStorage,
                        location='files')
model_args.add_argument(get_file_key(FileTypes.DATA), required=False, type=werkzeug.datastructures.FileStorage,
                        location='files')
model_args.add_argument('data_id', required=False, location='values')
model_args.add_argument('model_type', required=False, location='values')
model_args.add_argument('columns', required=False, location='values')
model_args.add_argument('parameters', required=False, location='values')
model_args.add_argument('separator', required=False, location='values')
model_args.add_argument('interval', required=False, location='values')
model_args.add_argument('prediction_interval', required=False, location='values', type=int)


class ModelController(Resource):
    def post(self):
        """
        Create a new model
        :return:
        """
        args = model_args.parse_args()
        data_file = args.get(get_file_key(FileTypes.DATA))
        separator = args.get("separator")
        data_id = args.get("data_id")
        model_type = args.get('model_type')
        columns = args.get('columns')
        parameters = args.get('parameters')
        interval = args.get('interval')
        if parameters is None:
            parameters = {}
        else:
            parameters = json.loads(parameters)

        if model_type is None:
            abort(400, message=f"The parameter 'model_type' is missing")
        columns = convert_columns_from_string(columns)

        if data_file is not None:
            model_id, errors, params = modelService.create_model_with_file(model_type, data_file, columns, parameters, separator, interval)
        elif data_id is not None:
            model_id, errors, params = modelService.create_model(model_type, data_id, columns, parameters, interval)
        else:
            current_app.logger.error("No data file or file id given")
            return {"model_id": None, "rmse": None}

        return {
            "model_id": model_id,
            "errors": errors,
            "parameters": params
        }

    def put(self):
        """
        Upload a model
        :return:
        """
        args = model_args.parse_args()

        model_file = args.get(get_file_key(FileTypes.MODEL))
        model_type = args.get('model_type')
        num_classes = args.get('num_classes')
        resolution = args.get('resolution')
        rmse = args.get('rmse')

        model = save_file(model_file, FileTypes.MODEL, {'model_type': model_type,
                                                        'num_classes': num_classes,
                                                        'resolution': resolution,
                                                        'rmse': rmse
                                                        })

        return {"model_id": model.id}


class ModelControllerId(Resource):
    def get(self, model_id):
        """
        Download a model
        :param model_id:
        :return:
        """
        return modelService.get_model(model_id)

    def delete(self, model_id):
        """
        delete a model
        :param model_id: The model to delete
        :return:
        """
        modelService.delete_model(model_id)

        return {"delete": model_id}


class ModelControllerIdForecast(Resource):
    def get(self, model_id):
        """
        Make a prediction
        :param model_id:
        :return:
        """
        args = model_args.parse_args()
        data_file = args.get(get_file_key(FileTypes.DATA))
        prediction_interval = args.get('prediction_interval')

        dataset = get_prediction_dataset(data_file)

        return modelService.create_forecast(model_id, prediction_interval, dataset)

    # For the showcase, becasue browsers do not allow bodies at get methods
    # will be fixed in api/v2
    def put(self, model_id):
        """
        Make a prediction
        :param model_id:
        :return:
        """
        args = model_args.parse_args()
        data_file = args.get(get_file_key(FileTypes.DATA))
        prediction_interval = args.get('prediction_interval')

        dataset = get_prediction_dataset(data_file)

        return modelService.create_forecast(model_id, prediction_interval, dataset)

