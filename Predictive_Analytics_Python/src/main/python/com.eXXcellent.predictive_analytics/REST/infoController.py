from flask import current_app, jsonify, Response

from flask_restful import Resource

from model.enums import InfoActionsEnum
from core.services.infoService import get_available_models, get_available_multivariant_models


class Info(Resource):

    def get(self, action):
        if action == InfoActionsEnum.MODELS.value:
            current_app.logger.info(f"GET request for {InfoActionsEnum.MODELS.value}")
            models = get_available_models()
            current_app.logger.info(f"Found models: {models}")

            result = {
                'models': models
            }

            return jsonify(result)
        elif action == InfoActionsEnum.MULTI_MODELS.value:
            current_app.logger.info(f"GET request for {InfoActionsEnum.MULTI_MODELS.value}")
            models = get_available_multivariant_models()
            current_app.logger.info(f"Found models: {models}")

            result = {
                'models': models
            }

            return jsonify(result)
