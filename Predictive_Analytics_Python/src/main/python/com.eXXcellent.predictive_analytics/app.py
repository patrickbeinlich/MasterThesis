from flask import Flask
from flask_cors import CORS
from flask_restful import Api
import logging

from db.database import db
from REST import dataController, modelController, infoController, autoController, debugController, fileController

import argparse
import config
import sys
import os

sys.path.append("../../../../models")

app = Flask(__name__)
# change to config.ProductionConfig for deployment
app.config.from_object(config.ProductionConfig)

try:
    app.config.from_prefixed_env()
except AttributeError:
    app.logger.info('no prefixed environment')

api = Api(app)

if app.config["DEBUG"]:
    # allow usage of cross-domain requests (both server and client on same device with same address {localhost})
    CORS(app)
    logging.basicConfig(level=logging.INFO,
                        format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
else:
    logging.basicConfig(filename=app.config["LOG_FILE"],
                        level=logging.INFO,
                        format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

db.app = app
db.init_app(app)

if not os.path.exists('./database.db'):
    db.create_all()
    app.logger.info('Database created')

global_path = '/api/v1'

model_path = global_path + '/models'
data_path = global_path + '/data'
file_request_path = global_path + '/files'
info_path = global_path + '/info'
automatic_path = global_path + '/auto'
debug_path = global_path + '/debug'

api.add_resource(modelController.ModelController,
                 model_path)
api.add_resource(modelController.ModelControllerId,
                 model_path + "/<int:model_id>")

api.add_resource(modelController.ModelControllerIdForecast,
                 model_path + "/<int:model_id>/prediction")

api.add_resource(dataController.DataController,
                 data_path + "/<string:action>")
api.add_resource(dataController.DataControllerId,
                 data_path + "/<string:action>/<int:file_id>")

api.add_resource(fileController.FileController,
                 file_request_path)
api.add_resource(fileController.FileControllerId,
                 file_request_path + "/<int:file_id>")

api.add_resource(infoController.Info,
                 info_path + "/<string:action>")

api.add_resource(autoController.AutomaticSelection,
                 automatic_path)
api.add_resource(autoController.AutomaticSelectionId,
                 automatic_path + "/<int:file_id>")


# api.add_resource(debugController.DebugController, debug_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="predictive analytics framework")
    parser.add_argument("-r", "--reset", help="reset database and other things", action='store_true', default=False, required=False)
    args = parser.parse_args()
    reset = args.reset

    if reset:
        file_folder = app.config["UPLOAD_FOLDER"]
        model_folder = app.config["MODEL_FOLDER"]
        for f in os.listdir(file_folder):
            os.remove(os.path.join(file_folder, f))
        for f in os.listdir(model_folder):
            os.remove(os.path.join(model_folder, f))

        db.drop_all()
        db.create_all()
        app.logger.info("Application reset")

    app.run(host=app.config["HOST"], port=app.config["PORT"], threaded=True)
