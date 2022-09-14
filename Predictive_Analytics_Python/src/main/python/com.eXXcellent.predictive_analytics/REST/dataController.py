import werkzeug.datastructures
from flask_restful import Resource, reqparse

import core.services.dataService as dataService

from core.services.fileService import get_file_key, FileTypes

data_args = reqparse.RequestParser()
data_args.add_argument(get_file_key(FileTypes.DATA), required=False, type=werkzeug.datastructures.FileStorage,
                       location='files')
data_args.add_argument('separator', required=False, location='values')
data_args.add_argument('interval', required=False, location='values')
data_args.add_argument('missing_dates', required=False, location='values')
data_args.add_argument('interpolation_method', required=False, location='values')
data_args.add_argument('find_missing', required=False, location='values')
data_args.add_argument('find_seasonality', required=False, location='values')
data_args.add_argument('decompose', required=False, location='values')


class DataControllerId(Resource):

    def post(self, action, file_id):
        args = data_args.parse_args()

        return dataService.execute_data_action_id(action, file_id, args)


class DataController(Resource):

    def post(self, action):
        args = data_args.parse_args()

        data_file = args.get(get_file_key(FileTypes.DATA))
        separator = args.get('separator')

        return dataService.execute_data_action_upload(action, data_file, separator, args)






