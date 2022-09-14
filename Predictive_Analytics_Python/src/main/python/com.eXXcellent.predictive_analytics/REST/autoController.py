import werkzeug.datastructures

from flask_restful import Resource, reqparse

from core.services.fileService import get_file_key, FileTypes, save_file, get_prediction_dataset
from core.services.automaticModelSelectionService import auto_model_selection
from core.services.dataPreparationService import convert_columns_from_string

auto_args = reqparse.RequestParser()
auto_args.add_argument(get_file_key(FileTypes.DATA), required=False, type=werkzeug.datastructures.FileStorage,
                       location='files')
auto_args.add_argument('separator', required=False, location='values')
auto_args.add_argument('interval', required=True, location='values')
auto_args.add_argument('columns', required=False, location='values')
auto_args.add_argument('models', required=False, location='values')
auto_args.add_argument('prediction_interval', required=True, location='values')
auto_args.add_argument('prediction_data', required=False, location='files', type=werkzeug.datastructures.FileStorage)


def select_model(args, file_id):
    interval = args.get('interval')
    columns = args.get('columns')
    models = args.get('models')
    prediction_interval = args.get('prediction_interval')
    pred_file = args.get('prediction_data')

    columns = convert_columns_from_string(columns)

    pred_dataset = get_prediction_dataset(pred_file)

    return auto_model_selection(file_id, interval, columns, models, prediction_interval, pred_dataset)


class AutomaticSelection(Resource):

    def post(self):
        args = auto_args.parse_args()

        data_file = args.get(get_file_key(FileTypes.DATA))
        separator = args.get('separator')
        file = save_file(data_file, FileTypes.DATA, {'separator': separator})
        return select_model(args, file.id)


class AutomaticSelectionId(Resource):

    def post(self, file_id):
        args = auto_args.parse_args()
        return select_model(args, file_id)
