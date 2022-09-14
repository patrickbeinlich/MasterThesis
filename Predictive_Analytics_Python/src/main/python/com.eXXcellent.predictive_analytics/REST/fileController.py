import werkzeug.datastructures

from flask import send_file
from flask_restful import Resource, reqparse


from core.services.fileService import save_file, FileTypes, get_file_key, get_data_file


file_args = reqparse.RequestParser()
file_args.add_argument(get_file_key(FileTypes.DATA), required=False, type=werkzeug.datastructures.FileStorage,
                        location='files')


class FileController(Resource):
    def put(self):
        args = file_args.parse_args()
        data_file = args.get(get_file_key(FileTypes.DATA))
        separator = args.get("separator")

        file = save_file(data_file, FileTypes.DATA, {'separator': separator})

        return {
            'file_id': file.id
        }


class FileControllerId(Resource):
    def get(self, file_id):
        data_file = get_data_file(file_id)

        return send_file(data_file.filePath, as_attachment=True)
